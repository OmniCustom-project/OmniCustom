import os
import time
import sys
import uuid
import cv2
import glob
import math
import numpy as np
import torch
import logging
import librosa
from typing import Union, Any
from textwrap import indent
import torch.nn as nn
from diffusers import FluxPipeline
from tqdm import tqdm
from distributed_comms.parallel_states import get_sequence_parallel_state, nccl_info
from utils.model_loading_utils import init_fusion_score_model_ovi, init_text_model, init_mmaudio_vae, init_wan_vae_2_2, load_fusion_checkpoint, load_fusion_lora
from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
import traceback
from omegaconf import OmegaConf
from utils.processing_utils import clean_text, preprocess_image_tensor, preprocess_audio_tensor, snap_hw_to_multiple_of_32, scale_hw_to_area_divisible
from modules.face_cropper.cropper import Cropper
from modules.face_cropper.crop_config import CropConfig
# face embedder
from insightface.app import FaceAnalysis
# audio embedder
from modules.ns3_codec.speaker_extractor import SpeakerExtractor

from optimum.quanto import freeze, qint8, quantize

DEFAULT_CONFIG = OmegaConf.load('./configs/inference/inference_fusion.yaml')

class OviFusionEngine:
    def __init__(self, config=DEFAULT_CONFIG, device=0, target_dtype=torch.bfloat16, self_lora=False):
        # Load fusion model
        self.device = device
        self.target_dtype = target_dtype
        meta_init = False
        self.cpu_offload = config.get("cpu_offload", False) or config.get("mode") == "t2i2v"
        self.crop_face = config.get("crop_face", False)
        if self.cpu_offload:
            logging.info("CPU offloading is enabled. Initializing all models aside from VAEs on CPU")

        model, video_config, audio_config = init_fusion_score_model_ovi(rank=device, meta_init=meta_init)
        
        face_ip_emb_dim = config.get("face_ip_emb_dim", 512)
        self.use_face_ip_emb = face_ip_emb_dim is not None
        audio_ip_emb_dim = config.get("audio_ip_emb_dim", 256)
        self.use_audio_ip_emb = audio_ip_emb_dim is not None
        
        # init lora
        if meta_init:
            with torch.device('meta'):
                model.init_lora(
                    self_lora=self_lora, 
                    train=False, 
                    vid_ip_emb_dim=face_ip_emb_dim,
                    audio_ip_emb_dim=audio_ip_emb_dim,
                )
        else:
            model.init_lora(
                self_lora=self_lora, 
                train=False,
                vid_ip_emb_dim=face_ip_emb_dim,
                audio_ip_emb_dim=audio_ip_emb_dim,
            )

        fp8 = config.get("fp8", False)
        int8 = config.get("qint8", False)
        if fp8:
            assert not config.get("mode") == "t2i2v", "Image generation with FluxPipeline is not supported with fp8 quantization. This is because if you are unable to run the bf16 model, you likely cannot run image gen model"

        if not meta_init:
            if not fp8:
                model = model.to(dtype=target_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu").eval()

        # Load VAEs
        vae_model_video = init_wan_vae_2_2(config.ckpt_dir, rank=device)
        vae_model_video.model.requires_grad_(False).eval()
        vae_model_video.model = vae_model_video.model.bfloat16()
        self.vae_model_video = vae_model_video

        vae_model_audio = init_mmaudio_vae(config.ckpt_dir, rank=device)
        vae_model_audio.requires_grad_(False).eval()
        self.vae_model_audio = vae_model_audio.bfloat16()

        # Load T5 text model
        self.text_model = init_text_model(config.ckpt_dir, rank=device, cpu_offload=self.cpu_offload)
        if config.get("shard_text_model", False):
            raise NotImplementedError("Sharding text model is not implemented yet.")
        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)

        # Find fusion ckpt in the same dir used by other components
        checkpoint_path = os.path.join(config.ckpt_dir, config.ckpt_name)

        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"No fusion checkpoint found in {config.ckpt_dir}")

        if meta_init:
            if not fp8:
                model = model.to(dtype=target_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu").eval()
            model.set_rope_params()
            
        load_fusion_checkpoint(model, checkpoint_path=checkpoint_path, from_meta=meta_init)
        # load lora weights
        lora_path = config.lora_path
        load_fusion_lora(model, ckpt_path=lora_path)
        
        self.model = model
        if int8:
            quantize(self.model, qint8)
            freeze(self.model)

        ## Load t2i as part of pipeline
        self.image_model = None
        
        if config.get("mode") == "t2i2v":
            logging.info(f"Loading Flux Krea for first frame generation...")
            self.image_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
            self.image_model.enable_model_cpu_offload(gpu_id=self.device) #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU VRAM

        # Fixed attributes, non-configurable
        self.audio_latent_channel = audio_config.get("in_dim")
        self.video_latent_channel = video_config.get("in_dim")
        self.audio_latent_length = 314 if '10s' in config.ckpt_name else 157              
        self.video_latent_length = 61 if '10s' in config.ckpt_name else 31
        
        # face cropper
        if self.crop_face:
            self.cropper: Cropper = Cropper(crop_cfg=CropConfig, device_id=torch.cuda.current_device())
        else:
            self.cropper = None
        
        # face embedder
        cuda_provider_options = {
            'device_id': self.device,  # 指定 GPU 设备 ID，默认为 0
            'arena_extend_strategy': 'kNextPowerOfTwo', # 内存分配策略
            'cudnn_conv_algo_search': 'EXHAUSTIVE', # 卷积算法搜索
            'do_copy_in_default_stream': True,
        }
        
        face_embedder_ckpt_dir = config.get("face_embedder_ckpt_dir", "./ckpts/InsightFace")
        self.face_embedder: FaceAnalysis = FaceAnalysis(name='antelopev2', root=face_embedder_ckpt_dir, providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider'])
        self.face_embedder.prepare(ctx_id=self.device, det_size=(640, 640))
        # warm up
        tic = time.time()
        img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        self.face_embedder.get(img_bgr)
        print(f"Warming up `face embedder, costs: {time.time() - tic} secs.")
        
        # audio embedder
        self.speaker_extractor = SpeakerExtractor(
            ckpt_path=config.get("audio_embedder_ckpt_dir", "../weights/naturalspeech3_facodec"), 
            device=self.device,
            dtype=self.target_dtype
        )
        
        logging.info(f"OVI Fusion Engine initialized, cpu_offload={self.cpu_offload}. GPU VRAM allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB, reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")
            
    def crop_image(self, cropper: Cropper, image_path: Union[str, np.ndarray], crop_size: Union[int, None]=None):
        """ image, RGB, HxWx3, [0, 255]
        """
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            assert isinstance(image_path, np.ndarray)
            image = image_path 
            
        try:
            cropped_image = cropper.crop_source_image(image, self.cropper.crop_cfg, crop_size=crop_size)["img_crop"] # default size 512
            #print(f"cropped_image shape: {cropped_image.shape}, value range: [{cropped_image.min()}, {cropped_image.max()}]")
        except Exception as e:
            print(f"Error <{e}> occurrs during crop images.")
            cropped_image = image
            
        return cropped_image

    def get_face_emb(self, image_path: Union[str, np.ndarray]):
        """ image, BGR, HxWx3, [0, 255]
        """
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            assert isinstance(image_path, np.ndarray)
            image = image_path
            image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        face_info = self.face_embedder.get(image)
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        face_emb = torch.from_numpy(face_info['embedding']).unsqueeze(0).to(device=self.device, dtype=self.target_dtype)
        return face_emb
    
    def get_audio_emb(self, audio_path: Union[str, np.ndarray]):
        """ audio, 1x1xL, wo norm
        """
        if isinstance(audio_path, str):
            audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)            
        else:
            assert isinstance(audio_path, np.ndarray)
            audio = audio_path
        audio = audio[:64000]
        audio = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
        
        return self.speaker_extractor(audio.to(dtype=self.target_dtype, device=self.device)).to(device=self.device, dtype=self.target_dtype)
                    
    @torch.inference_mode()
    def generate(self,
                    text_prompt, 
                    image_path=None,
                    ip_image_path=None,
                    ip_audio_path=None,
                    video_frame_height_width=None,
                    seed=100,
                    solver_name="unipc",
                    sample_steps=50,
                    shift=5.0,
                    video_guidance_scale=5.0,
                    audio_guidance_scale=4.0,
                    slg_layer=9,
                    video_negative_prompt="",
                    audio_negative_prompt=""
                ):

        params = {
            "Text Prompt": text_prompt,
            "Image Path": image_path if image_path else "None (T2V mode)",
            "IP Image Path": ip_image_path if ip_image_path else "None",
            "IP Audio Path": ip_audio_path if ip_audio_path else "None",
            "Crop Face": self.crop_face and self.cropper is not None,
            "Frame Height Width": video_frame_height_width,
            "Seed": seed,
            "Solver": solver_name,
            "Sample Steps": sample_steps,
            "Shift": shift,
            "Video Guidance Scale": video_guidance_scale,
            "Audio Guidance Scale": audio_guidance_scale,
            "SLG Layer": slg_layer,
            "Video Negative Prompt": video_negative_prompt,
            "Audio Negative Prompt": audio_negative_prompt,
        }

        pretty = "\n".join(f"{k:>24}: {v}" for k, v in params.items())
        logging.info("\n========== Generation Parameters ==========\n"
                    f"{pretty}\n"
                    "==========================================")
        try:
            scheduler_video, timesteps_video = self.get_scheduler_time_steps(
                sampling_steps=sample_steps,
                device=self.device,
                solver_name=solver_name,
                shift=shift
            )
            scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
                sampling_steps=sample_steps,
                device=self.device,
                solver_name=solver_name,
                shift=shift
            )
            is_id2v = (ip_image_path is not None) or (ip_audio_path is not None)
            is_t2v = image_path is None and not is_id2v
            is_i2v = not is_t2v and not is_id2v
            
            ip_image, ip_audio = None, None
            face_emb, audio_emb = None, None
            if ip_image_path:
                if self.crop_face and self.cropper is not None:
                    assert video_frame_height_width is not None, f"If mode=id2v, video_frame_height_width must be provided."
                    infer_h, infer_w = video_frame_height_width
                    train_h, train_w = 480, 864
                    train_crop_size = 512
                    infer_crop_size = int(math.sqrt((infer_h * infer_w) / (train_h * train_w)) * train_crop_size)
                    infer_crop_size = (infer_crop_size // 16) * 16   # compateble with vae2.2
                    ip_image = preprocess_image_tensor(self.crop_image(self.cropper, ip_image_path, crop_size=infer_crop_size), self.device, self.target_dtype, resize_total_area=512*512)
                else:
                    ip_image = preprocess_image_tensor(ip_image_path, self.device, self.target_dtype, resize_total_area=512*512)
                print(f"ip_image shape: {ip_image.shape}")
                # get face emb
                if self.use_face_ip_emb:
                    face_emb = self.get_face_emb(ip_image_path)
                    print(f"face embedding shape: {face_emb.shape}")
            if ip_audio_path:
                ip_audio = preprocess_audio_tensor(ip_audio_path, self.device, self.target_dtype, clip_len=64000)
                print(f"ip_audio shape: {ip_audio.shape}")
                # get audio emb
                if self.use_audio_ip_emb:
                    audio_emb = self.get_audio_emb(ip_audio_path)
                    print(f"audio embedding shape: {audio_emb.shape}")
                
            first_frame = None
            image = None
            if is_i2v and not self.image_model:
                # Load first frame from path
                first_frame = preprocess_image_tensor(image_path, self.device, self.target_dtype)
            else:   
                assert video_frame_height_width is not None, f"If mode=t2v or t2i2v or id2v, video_frame_height_width must be provided."
                video_h, video_w = video_frame_height_width
                snap_area = max(video_h * video_w, 720 * 720)
                video_h, video_w = snap_hw_to_multiple_of_32(video_h, video_w, area = snap_area)
                video_latent_h, video_latent_w = video_h // 16, video_w // 16       
                if self.image_model is not None:
                    # this already means t2v mode with image model
                    image_h, image_w = scale_hw_to_area_divisible(video_h, video_w, area = 1024 * 1024)
                    image = self.image_model(
                        clean_text(text_prompt),
                        height=image_h,
                        width=image_w,
                        guidance_scale=4.5,
                        generator=torch.Generator().manual_seed(seed)
                    ).images[0]
                    first_frame = preprocess_image_tensor(image, self.device, self.target_dtype)
                    is_i2v = True
                else:
                    print(f"Pure T2V mode: calculated video latent size: {video_latent_h} x {video_latent_w}")
            
            if self.cpu_offload:
                self.text_model.model = self.text_model.model.to(self.device)
            text_embeddings = self.text_model([text_prompt, video_negative_prompt, audio_negative_prompt], self.text_model.device)
            text_embeddings = [emb.to(self.target_dtype).to(self.device) for emb in text_embeddings]

            if self.cpu_offload:
                self.offload_to_cpu(self.text_model.model)

            # Split embeddings
            text_embeddings_audio_pos = text_embeddings[0]
            text_embeddings_video_pos = text_embeddings[0] 

            text_embeddings_video_neg = text_embeddings[1]
            text_embeddings_audio_neg = text_embeddings[2]

            if is_i2v:
                if self.cpu_offload:
                    self.vae_model_video.model = self.vae_model_video.model.to(
                        self.device
                    )
                with torch.no_grad():
                    latents_images = self.vae_model_video.wrapped_encode(first_frame[:, :, None]).to(self.target_dtype).squeeze(0) # c 1 h w 
                latents_images = latents_images.to(self.target_dtype)
                video_latent_h, video_latent_w = latents_images.shape[2], latents_images.shape[3]
                if self.cpu_offload:
                    self.offload_to_cpu(self.vae_model_video.model)
            
            if ip_image is not None:
                with torch.no_grad():
                    print(f"ip image shape before VAE {ip_image.shape}")
                    ip_image = self.vae_model_video.wrapped_encode(ip_image[:, :, None]).to(self.target_dtype) # 1 c 1 h w 
                    print(f"ip image shape after VAE {ip_image.shape}")
                
            if ip_audio is not None:
                print(f"ip audio shape before VAE {ip_audio.shape}")
                ip_audio = self.vae_model_audio.wrapped_encode(ip_audio).to(dtype=self.target_dtype, device=self.device).transpose(-1, -2)
                print(f"ip audio shape after VAE {ip_audio.shape}")

            video_noise = torch.randn((self.video_latent_channel, self.video_latent_length, video_latent_h, video_latent_w), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # c, f, h, w
            audio_noise = torch.randn((self.audio_latent_length, self.audio_latent_channel), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # 1, l c -> l, c
            
            # Calculate sequence lengths from actual latents
            max_seq_len_audio = audio_noise.shape[0]  # L dimension from latents_audios shape [1, L, D]
            _patch_size_h, _patch_size_w = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
            max_seq_len_video = video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (_patch_size_h*_patch_size_w) # f * h * w from [1, c, f, h, w]
            
            # Sampling loop
            if self.cpu_offload:
                self.offload_to_cpu(self.vae_model_video.model)
                self.offload_to_cpu(self.vae_model_audio)
                self.model = self.model.to(self.device)
            with torch.amp.autocast('cuda', enabled=self.target_dtype != torch.float32, dtype=self.target_dtype):
                for i, (t_v, t_a) in tqdm(enumerate(zip(timesteps_video, timesteps_audio))):
                    timestep_input = torch.full((1,), t_v, device=self.device)

                    if is_i2v:
                        video_noise[:, :1] = latents_images

                    # Positive (conditional) forward pass
                    pos_forward_args = {
                        'audio_context': [text_embeddings_audio_pos],
                        'vid_context': [text_embeddings_video_pos],
                        'vid_seq_len': max_seq_len_video,
                        'audio_seq_len': max_seq_len_audio,
                        'first_frame_is_clean': is_i2v
                    }

                    pred_vid_pos, pred_audio_pos = self.model(
                        vid=[video_noise],
                        audio=[audio_noise],
                        t=timestep_input,
                        vid_ip=ip_image,
                        audio_ip=ip_audio,
                        vid_ip_emb=face_emb,
                        audio_ip_emb=audio_emb,
                        **pos_forward_args
                    )
                    
                    # Negative (unconditional) forward pass  
                    neg_forward_args = {
                        'audio_context': [text_embeddings_audio_neg],
                        'vid_context': [text_embeddings_video_neg],
                        'vid_seq_len': max_seq_len_video,
                        'audio_seq_len': max_seq_len_audio,
                        'first_frame_is_clean': is_i2v,
                        'slg_layer': slg_layer
                    }
                    
                    pred_vid_neg, pred_audio_neg = self.model(
                        vid=[video_noise],
                        audio=[audio_noise],
                        t=timestep_input,
                        vid_ip=ip_image,
                        audio_ip=ip_audio,
                        vid_ip_emb=face_emb,
                        audio_ip_emb=audio_emb,
                        **neg_forward_args
                    )

                    # Apply classifier-free guidance
                    pred_video_guided = pred_vid_neg[0] + video_guidance_scale * (pred_vid_pos[0] - pred_vid_neg[0])
                    pred_audio_guided = pred_audio_neg[0] + audio_guidance_scale * (pred_audio_pos[0] - pred_audio_neg[0])

                    # Update noise using scheduler
                    video_noise = scheduler_video.step(
                        pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                    audio_noise = scheduler_audio.step(
                        pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                if self.cpu_offload:
                    self.offload_to_cpu(self.model)
                    self.vae_model_video.model = self.vae_model_video.model.to(
                        self.device
                    )
                    self.vae_model_audio = self.vae_model_audio.to(self.device)

                if is_i2v:
                    video_noise[:, :1] = latents_images

                # Decode audio
                audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2)  # 1, c, l
                generated_audio = self.vae_model_audio.wrapped_decode(audio_latents_for_vae)
                generated_audio = generated_audio.squeeze().cpu().float().numpy()
                
                # Decode video  
                video_latents_for_vae = video_noise.unsqueeze(0)  # 1, c, f, h, w
                generated_video = self.vae_model_video.wrapped_decode(video_latents_for_vae)
                generated_video = generated_video.squeeze(0).cpu().float().numpy()  # c, f, h, w
                if self.cpu_offload:
                    self.offload_to_cpu(self.vae_model_video.model)
                    self.offload_to_cpu(self.vae_model_audio)
            
            return generated_video, generated_audio, image


        except Exception as e:
            logging.error(traceback.format_exc())
            return None
            
    def offload_to_cpu(self, model):
        model = model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return model

    def get_scheduler_time_steps(self, sampling_steps, solver_name='unipc', device=0, shift=5.0):
        torch.manual_seed(4)

        if solver_name == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps

        elif solver_name == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)
            
        elif solver_name == 'euler':
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                shift=shift
            )
            timesteps, sampling_steps = retrieve_timesteps(
                sample_scheduler,
                sampling_steps,
                device=device,
            )
        
        else:
            raise NotImplementedError("Unsupported solver.")
        
        return sample_scheduler, timesteps
