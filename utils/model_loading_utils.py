import torch 
import os
import json
from safetensors.torch import load_file

from modules.fusion import FusionModel
from modules.t5 import T5EncoderModel
from modules.vae2_2 import Wan2_2_VAE
from modules.mmaudio.features_utils import FeaturesUtils
    
def init_wan_vae_2_2(ckpt_dir, rank=0):
    vae_config = {}
    vae_config['device'] = rank
    vae_pth = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    vae_config['vae_pth'] = vae_pth
    vae_model = Wan2_2_VAE(**vae_config)

    return vae_model

def init_mmaudio_vae(ckpt_dir, rank=0):
    vae_config = {}
    vae_config['mode'] = '16k'
    vae_config['need_vae_encoder'] = True

    tod_vae_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/v1-16.pth")
    bigvgan_vocoder_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/best_netG.pt")

    vae_config['tod_vae_ckpt'] = tod_vae_ckpt
    vae_config['bigvgan_vocoder_ckpt'] = bigvgan_vocoder_ckpt

    vae = FeaturesUtils(**vae_config).to(rank)

    return vae

def init_fusion_score_model_ovi(rank: int = 0, meta_init=False):
    video_config = "configs/model/dit/video.json"
    audio_config = "configs/model/dit/audio.json"
    assert os.path.exists(video_config), f"{video_config} does not exist"
    assert os.path.exists(audio_config), f"{audio_config} does not exist"

    with open(video_config) as f:
        video_config = json.load(f)

    with open(audio_config) as f:
        audio_config = json.load(f)

    if meta_init:
        with torch.device("meta"):
            fusion_model = FusionModel(video_config, audio_config)
    else:
        fusion_model = FusionModel(video_config, audio_config)
    
    params_all = sum(p.numel() for p in fusion_model.parameters())
    
    if rank == 0:
        print(
            f"Score model (Fusion) all parameters:{params_all}"
        )

    return fusion_model, video_config, audio_config

def init_text_model(ckpt_dir, rank, cpu_offload=False):
    wan_dir = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B")
    text_encoder_path = os.path.join(wan_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    text_tokenizer_path = os.path.join(wan_dir, "google/umt5-xxl")

    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=rank,
        checkpoint_path=text_encoder_path,
        tokenizer_path=text_tokenizer_path,
        cpu_offload=cpu_offload,
        shard_fn=None)

    return text_encoder


def load_fusion_checkpoint(model, checkpoint_path, from_meta=False, strict=False):
    if checkpoint_path and os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".safetensors"): 
            df = load_file(checkpoint_path, device="cpu")
        elif checkpoint_path.endswith(".pt"):
            try:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                df = df['module'] if 'module' in df else df
            except Exception as e:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                df = df['app']['model']
        else: 
            raise RuntimeError("We only support .safetensors and .pt checkpoints")

        missing, unexpected = model.load_state_dict(df, strict=strict, assign=from_meta)
        #print(f"Missing Keys: [{missing}]")
        #print(f"Unexpected Keys: [{unexpected}]")
        
        del df
        import gc
        gc.collect()
        print(f"Successfully loaded fusion checkpoint from {checkpoint_path}")
    else: 
        raise RuntimeError("{checkpoint=} does not exists'")
    
    
def load_fusion_lora(fusion, ckpt_path, from_meta=False, strict=True):
    print("=" * 45 + " Loading LoRA Weights " + "=" * 45)
    if ckpt_path and os.path.exists(ckpt_path):
        
        if ckpt_path.endswith(".safetensors"): 
            df = load_file(ckpt_path, device="cpu")
        elif ckpt_path.endswith(".pt"):
            try:
                df = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                df = df['module'] if 'module' in df else df
            except Exception as e:
                df = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                df = df['app']['model']
        else: 
            raise RuntimeError("We only support .safetensors and .pt checkpoints")
        
        state_dict = df.get("state_dict", df)
        #print(state_dict.keys())
        #print(f"=" * 90)
        #print(state_dict.keys())
        
        model = {}
        # audio model
        if hasattr(fusion.audio_model, "ip_projection"):
            print(f"[Audio Model] Loading IP_PROJECTION")
            for sub in ["0", "2"]:
                weight_key = f"audio_model.ip_projection.{sub}.weight"
                bias_key = f"audio_model.ip_projection.{sub}.bias"
                if hasattr(getattr(fusion.audio_model, "ip_projection"), sub):
                    model[weight_key] = getattr(getattr(fusion.audio_model, "ip_projection"), sub).weight
                    model[bias_key] = getattr(getattr(fusion.audio_model, "ip_projection"), sub).bias
                else:
                    if strict:
                        raise KeyError(f"Missing module: {key}")
        print(f"[Audio Model] Loading LoRAs & IP_EMBEDDING Layer")
        for i, block in enumerate(fusion.audio_model.blocks):
            prefix = f"audio_model.blocks.{i}.self_attn."
            attn = block.self_attn
            for name in ["q_loras", "k_loras", "v_loras", "o_loras", "s_q_loras", "s_k_loras", "s_v_loras", "s_o_loras"]:
                if hasattr(attn, name):
                    for sub in ["down", "up"]:
                        key = f"{prefix}{name}.{sub}.weight"
                        if hasattr(getattr(attn, name), sub):
                            model[key] = getattr(getattr(attn, name), sub).weight
                        else:
                            if strict:
                                raise KeyError(f"Missing module: {key}")
            # load ip embedding layer
            name = "ip_embedding"
            weight_key = f"{prefix}{name}.weight"
            bias = f"{prefix}{name}.bias"
            if hasattr(attn, name):
                model[weight_key] = getattr(attn, name).weight
                model[bias] = getattr(attn, name).bias
        
        # video model
        if hasattr(fusion.video_model, "ip_projection"):
            print(f"[Video Model] Loading IP_PROJECTION")
            for sub in ["0", "2"]:
                weight_key = f"video_model.ip_projection.{sub}.weight"
                bias_key = f"video_model.ip_projection.{sub}.bias"
                if hasattr(getattr(fusion.video_model, "ip_projection"), sub):
                    model[weight_key] = getattr(getattr(fusion.video_model, "ip_projection"), sub).weight
                    model[bias_key] = getattr(getattr(fusion.video_model, "ip_projection"), sub).bias
                else:
                    if strict:
                        raise KeyError(f"Missing module: {key}")
        print(f"[Video Model] Loading LoRAs & IP_EMBEDDING Layer")
        for i, block in enumerate(fusion.video_model.blocks):
            prefix = f"video_model.blocks.{i}.self_attn."
            attn = block.self_attn
            for name in ["q_loras", "k_loras", "v_loras", "o_loras", "s_q_loras", "s_k_loras", "s_v_loras", "s_o_loras"]:
                if hasattr(attn, name):
                    for sub in ["down", "up"]:
                        key = f"{prefix}{name}.{sub}.weight"
                        if hasattr(getattr(attn, name), sub):
                            model[key] = getattr(getattr(attn, name), sub).weight
                        else:
                            if strict:
                                raise KeyError(f"Missing module: {key}")
            # load ip embedding layer
            name = "ip_embedding"
            weight_key = f"{prefix}{name}.weight"
            bias = f"{prefix}{name}.bias"
            if hasattr(attn, name):
                model[weight_key] = getattr(attn, name).weight
                model[bias] = getattr(attn, name).bias

        for k, param in state_dict.items():
            if k in model:
                if model[k].shape != param.shape:
                    if strict:
                        raise ValueError(
                            f"Shape mismatch: {k} | {model[k].shape} vs {param.shape}"
                        )
                    else:
                        continue
                model[k].data.copy_(param)
            else:
                if strict and "pipe.speaker_extractor" not in k:
                    raise KeyError(f"Unexpected key in ckpt: {k}")
    else: 
        raise RuntimeError("{checkpoint=} does not exists'")
    
    print("=" * 45 + " Loading LoRA Weights " + "=" * 45)