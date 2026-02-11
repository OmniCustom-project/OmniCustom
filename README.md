# OmniCustom: Sync Audio-Video Customization Via Joint Audio-Video Generation Model

<div align="center">

[![project page](https://img.shields.io/badge/Project_page-More_visualizations-green)](https://OmniCustom-project.github.io/page/)&nbsp;
<a href="https://huggingface.co/bytedance-research/Phantom"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=orange"></a>

</div>

## 🔥 Latest News!

* Feb 14, 2025: We proposed **OmniCustom**, a novel framework to deal with sync audio-video customization.  For more video demos, please visit the [project page](https://OmniCustom-project.github.io/page/).


## 🎥 Video


https://github.com/user-attachments/assets/7943515a-691b-417e-99c7-65003a63e258


## 📖 Overview

Given a reference image $I^{r}$ and a reference audio $A^{r}$, our **OmniCustom** framework synchronously generates a video that preserves the visual identity from $I^{r}$ and an audio track that mimics the timbre of $A^{r}$. Here, the speech content can be freely specified through a textual prompt.

<p align="center">
  <img src="assets/images/teaser.png">
</p>

## ⚡️ Quickstart

### Installation

##### 1.Clone the repo:

```sh
git clone https://github.com/Phantom-video/Phantom.git
cd OmniCustom
```

##### 2. Create Environment:

```sh
conda create -n omnicustom python=3.10
conda activate omnicustom
pip install -r requirements.txt
```

##### 3. Install Flash Attention :

```sh
pip install flash-attn --no-build-isolation
```

### Model Download

| Models       | Download Link                                                                                                                                           |    Notes                      |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| OmniCustom models      | 🤗 [Huggingface](https://huggingface.co/bytedance-research/Phantom/blob/main/Phantom-Wan-1.3B.pth)   | 1.8G
| Naturalspeech 3 | 🤗 [Huggingface](https://huggingface.co/bytedance-research/Phantom/tree/main) | Timbre embedding extractor
|InsightFace | 🤗 [Huggingface](https://huggingface.co/bytedance-research/Phantom/tree/main) | Face embedding extractor

First you need to download the original model of OVI, Wan2.2-TI2V-5B, and MMAudio. You can download them using `download_weights.py`, and put them into `ckpts`:

```sh
python3 download_weights.py --output-dir ./ckpts
```

Then download the model of our OmniCustom, Naturalspeech 3, InsightFace from Huggingface:

```sh
pip install "huggingface_hub[cli]"
huggingface-cli download bytedance-research/Phantom --local-dir ./ckpts
huggingface-cli download bytedance-research/Phantom --local-dir ./ckpts
huggingface-cli download bytedance-research/Phantom --local-dir ./ckpts
```

## ⚙️ Configure OmniCustom

The configure file of  OmniCustom [OmniCustom/configs/inference/inference_fusion.yaml](OmniCustom/configs/inference/inference_fusion.yaml ) can be modified. The following parameters control generation quality, video resolution, and how text, image, and audio inputs are balanced:

```yaml
ckpt_name: Ovi/model.safetensors  #base model
lora_path: ./ckpts/step-92000.safetensors #the checkpoint of our OmniCustom
self_lora: true 
# face embedder 
face_embedder_ckpt_dir: ./ckpts/InsightFace  
face_ip_emb_dim: 512   
# audio embedder
audio_embedder_ckpt_dir: ./ckpts/naturalspeech3_facodec
audio_ip_emb_dim: 256 
# output
output_dir: ./outputs/
sample_steps: 50  # number of denoising steps. Lower (30-40) = faster generation
solver_name: unipc  # sampling algorithm for denoising process
shift: 5.0    #timestep shift factor for sampling scheduler
sp_size: 1
audio_guidance_scale: 3.0
video_guidance_scale: 4.0
mode: "id2v"                                                  # ["id2v", "t2v", "i2v", "t2i2v"] all comes with audio
fp8: False                        # load fp8 version of model, will have quality degradation and will not have speed 
cpu_offload: False
seed: 102                    # random seed for reproducible results
crop_face: true        # crop face region from the reference image
video_negative_prompt: "jitter, bad hands, blur, distortion, two people, two persons, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border" 
audio_negative_prompt: "robotic, muffled, echo, distorted"    # avoid artifacts in audio
video_frame_height_width: [576, 992] #[512, 992]                         # only useful if mode = t2v or t2i2v, recommended values: [512, 992], [992, 512], [960, 512], [512, 960], [720, 720], [448, 1120]
text_prompt: ./example_prompts/benchmark_example.csv  #group generation
slg_layer: 11
each_example_n_times: 1
```



## 🔑 Inference

##### Single GPU

```sh
bash inference.sh
```

> 💡Note:
> 
> * `text_prompt` in `configs/inference/inference_fusion.yaml` can change examples for sync audio-video customization.
> * Those results without any customization and those with only identity customization will be saved to the result folder.
> * When the generated video is unsatisfactory, the most straightforward solution is to try changing the `seed` in `configs/inference/inference_fusion.yaml`.
> * The Peak VRAM Required is 80 GB in a single GPU.





##### More Results
<table width="100%" border="1" cellpadding="20" cellspacing="0" align="center" style="border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
  <thead>
    <tr bgcolor="#f5f5f5">
      <th style="width:18%; padding:16px; border:1px solid #ddd; font-size:14px;">Reference Images</th>
      <th style="width:18%; padding:16px; border:1px solid #ddd; font-size:14px;">Reference Audios</th>
      <th style="width:24%; padding:16px; border:1px solid #ddd; font-size:14px;">Text prompts</th>
      <th style="width:40%; padding:16px; border:1px solid #ddd; font-size:14px;">Generated Videos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <!-- 行1 -->
      <td align="center" style="vertical-align:middle; padding:16px; border:1px solid #ddd; min-height:300px;">
        <img src="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/images/ref_68.png" alt="Reference Image 1" style="max-height:200px; max-width:100%; object-fit: contain;">
      </td>
      <td align="center" style="vertical-align:middle; padding:16px; border:1px solid #ddd; min-height:300px;">
        <div style="width:160px; margin:0 auto;">
          <audio controls preload="metadata" style="width:100%; max-height:40px;">
            <source src="https://OmniCustom-project.github.io/OmniCustom/assets/audio/00068_audio.wav" type="audio/wav">
          </audio>
          <br>
          <a href="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/audio/00068_audio.wav" download style="font-size:11px; color:#0366d6; text-decoration:none;">
            ⤓ Download WAV ~5s
          </a>
        </div>
      </td>
      <td style="line-height:1.6; vertical-align:middle; padding:16px; border:1px solid #ddd; font-size:13px; min-height:300px;">
        <div style="max-height:280px; overflow-y: auto; padding-right: 8px;">
          A man stands at the podium in OpenAI's luxurious conference room, behind him a massive electronic screen displays the company's glowing profit data. He grips the microphone firmly, gazes across the audience below, and announces in a steady tone: &lt;S&gt;The board wants to sell OpenAI to Zuckerberg, which is unacceptable.&lt;E&gt;
        </div>
      </td>
      <td align="center" style="vertical-align:middle; padding:16px; border:1px solid #ddd; min-height:300px;">
        <!-- 方法1：使用纯链接（最简单） -->
        <div style="text-align: center;">
          <a href="https://github.com/user-attachments/assets/5472ec3d-57bc-45b3-a921-7913d0bd8bb7" style="text-decoration:none; color:#0366d6; font-weight:500;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 24px; border-radius: 8px; display: inline-block; margin: 8px 0;">
              ▶️ Play Demo Video 1
            </div>
          </a>
          <br>
          <div style="margin-top: 8px; font-size: 12px; color: #666;">
            <a href="https://github.com/user-attachments/assets/5472ec3d-57bc-45b3-a921-7913d0bd8bb7" download style="color:#0366d6; text-decoration:none;">
              ⤓ Download MP4
            </a>
          </div>
        </div>
      </td>
    </tr>
    <tr>
      <!-- 行2 -->
      <td align="center" style="vertical-align:middle; padding:16px; border:1px solid #ddd; min-height:300px;">
        <img src="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/images/ref_69.png" alt="Reference Image 2" style="max-height:200px; max-width:100%; object-fit: contain;">
      </td>
      <td align="center" style="vertical-align:middle; padding:16px; border:1px solid #ddd; min-height:300px;">
        <div style="width:160px; margin:0 auto;">
          <audio controls preload="metadata" style="width:100%; max-height:40px;">
            <source src="https://OmniCustom-project.github.io/OmniCustom/assets/audio/00069_audio.wav" type="audio/wav">
          </audio>
          <br>
          <a href="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/audio/00069_audio.wav" download style="font-size:11px; color:#0366d6; text-decoration:none;">
            ⤓ Download WAV ~5s
          </a>
        </div>
      </td>
      <td style="line-height:1.6; vertical-align:middle; padding:16px; border:1px solid #ddd; font-size:13px; min-height:300px;">
        <div style="max-height:280px; overflow-y: auto; padding-right: 8px;">
          A woman stands before the iconic Rockefeller Center Christmas Tree, its thousands of lights reflecting in her eyes as snow begins to fall gently around her. Wearing a tartan scarf and holding a cup of steaming cocoa, she brings her mittened hands together and speaks softly into the frosty air: &lt;S&gt;May the spirit of Christmas fill your heart throughout the coming year.&lt;E&gt;
        </div>
      </td>
      <td align="center" style="vertical-align:middle; padding:16px; border:1px solid #ddd; min-height:300px;">
        <!-- 方法2：使用图片按钮样式 -->
        <div style="text-align: center;">
          <a href="https://github.com/user-attachments/assets/5472ec3d-57bc-45b3-a921-7913d0bd8bb7">
            <div style="background-color: #f6f8fa; border: 2px dashed #d0d7de; border-radius: 8px; padding: 16px; display: inline-block; min-width: 200px;">
              <div style="font-size: 24px; margin-bottom: 8px;">🎬</div>
              <div style="font-weight: 600; color: #24292f; margin-bottom: 4px;">Play Demo Video 2</div>
              <div style="font-size: 11px; color: #57606a;">Click to watch in browser</div>
            </div>
          </a>
          <div style="margin-top: 8px; font-size: 12px; color: #666;">
            <a href="https://github.com/user-attachments/assets/5472ec3d-57bc-45b3-a921-7913d0bd8bb7" download style="color:#0366d6; text-decoration:none; margin-right: 12px;">
              ⤓ Download
            </a>
            <span style="color: #8b949e;">MP4 format</span>
          </div>
        </div>
      </td>
    </tr>
    <tr>
      <!-- 行3 -->
      <td align="center" style="vertical-align:middle; padding:16px; border:1px solid #ddd; min-height:300px;">
        <img src="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/images/ref_70.png" alt="Reference Image 3" style="max-height:200px; max-width:100%; object-fit: contain;">
      </td>
      <td align="center" style="vertical-align:middle; padding:16px; border:1px solid #ddd; min-height:300px;">
        <div style="width:160px; margin:0 auto;">
          <audio controls preload="metadata" style="width:100%; max-height:40px;">
            <source src="https://OmniCustom-project.github.io/OmniCustom/assets/audio/00070_audio.wav" type="audio/wav">
          </audio>
          <br>
          <a href="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/audio/00070_audio.wav" download style="font-size:11px; color:#0366d6; text-decoration:none;">
            ⤓ Download WAV ~5s
          </a>
        </div>
      </td>
      <td style="line-height:1.6; vertical-align:middle; padding:16px; border:1px solid #ddd; font-size:13px; min-height:300px;">
        <div style="max-height:280px; overflow-y: auto; padding-right: 8px;">
          A man stands on a bustling street in Shanghai, the air thick with the festive atmosphere of Chinese Lunar New Year, with numerous red lanterns hanging in clusters overhead. He blends seamlessly into the vibrant surroundings, then clasps his hands together in a traditional gesture of greeting and says warmly: &lt;S&gt;Wishing everyone a Happy New Year and joy every single day.&lt;E&gt;
        </div>
      </td>
      <td align="center" style="vertical-align:middle; padding:16px; border:1px solid #ddd; min-height:300px;">
        <!-- 方法3：简洁链接 -->
        <div style="text-align: center;">
          <a href="https://github.com/user-attachments/assets/5472ec3d-57bc-45b3-a921-7913d0bd8bb7" style="display: inline-block; text-decoration: none;">
            <div style="display: flex; align-items: center; justify-content: center; background-color: #0969da; color: white; padding: 10px 20px; border-radius: 6px; font-weight: 500;">
              <svg style="width: 16px; height: 16px; margin-right: 8px;" fill="currentColor" viewBox="0 0 16 16">
                <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>
                <path d="M6.271 5.055a.5.5 0 0 1 .52.038l3.5 2.5a.5.5 0 0 1 0 .814l-3.5 2.5A.5.5 0 0 1 6 10.5v-5a.5.5 0 0 1 .271-.445z"/>
              </svg>
              Play Demo Video 3
            </div>
          </a>
          <div style="margin-top: 12px; font-size: 12px; color: #57606a;">
            <a href="https://github.com/user-attachments/assets/5472ec3d-57bc-45b3-a921-7913d0bd8bb7" download style="color:#0969da; text-decoration:none;">
              <span style="background-color: #f6f8fa; padding: 4px 8px; border-radius: 4px; border: 1px solid #d0d7de;">
                📥 Download Video
              </span>
            </a>
            <div style="margin-top: 6px; font-size: 11px;">High quality MP4 • 5-10 seconds</div>
          </div>
        </div>
      </td>
    </tr>
  </tbody>
</table>

<!-- 提示用户替换链接 -->
<div style="margin-top: 20px; padding: 12px; background-color: #f6f8fa; border-left: 4px solid #0969da; font-size: 13px; color: #24292f;">
  <strong>🔗 链接替换说明：</strong> 
  请为每个视频生成不同的GitHub附件链接，并替换上面的三个链接地址：
  <ul style="margin-top: 8px; margin-bottom: 0;">
    <li>Video 1: <code>https://github.com/user-attachments/assets/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx</code></li>
    <li>Video 2: <code>https://github.com/user-attachments/assets/yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy</code></li>
    <li>Video 3: <code>https://github.com/user-attachments/assets/zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz</code></li>
  </ul>
</div>


Please see our project page(https://OmniCustom-project.github.io/page/) for better audio presentation.

## 📑 Todo List

- [x] Inference Codes and Checkpoint of OmniCustom
- [ ] Open Source Evaluation Benchmark
- [ ] Open Source OmniCustom-1M dataset
- [ ] Training Codes of OmniCustom

## 🙏 Acknowledgements

We would like to thank the following projects:

**[OVI](https://github.com/character-ai/Ovi)**: Our OmniCustom is finetuned over OVI for ID and timbre customization.

**[Naturalspeech 3](https://github.com/lifeiteng/naturalspeech3_facodec)**: 256-D timbre embeddings are extracted using Naturalspeech 3.

**[InsightFace](https://github.com/deepinsight/insightface)**: 512-D face embeddings are extracted using InsightFace.

**[MMAudio](https://github.com/hkchengrex/MMAudio)**: Audio vae is provided by MMAudio.

**[Wan2.2](https://github.com/Wan-Video/Wan2.2)**: The video branch is initialized from the Wan2.2 repository.

## ⭐

If OmniCustom is helpful, please help to ⭐ the repo.

## 📚 Citation

We will link the paper soon.
