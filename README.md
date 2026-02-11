# OmniCustom: Sync Audio-Video Customization Via Joint Audio-Video Generation Model

<div align="center">

[![project page](https://img.shields.io/badge/Project_page-More_visualizations-green)](https://OmniCustom-project.github.io/page/)&nbsp;
<a href="https://huggingface.co/bytedance-research/Phantom"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=orange"></a>

</div>

## 🔥 Latest News!

* Feb 14, 2025: We proposed **OmniCustom**, a novel framework to deal with sync audio-video customization.  For more video demos, please visit the [project page](https://OmniCustom-project.github.io/page/).


## 🎥 Video

https://github.com/user-attachments/assets/9fd12b40-41ab-4201-8667-8b333db1123d

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



##### More examples

<table width="100%" border="1" cellpadding="12" cellspacing="0" align="center" style="border-collapse: collapse;">
  <thead>
    <tr bgcolor="#f5f5f5">
      <th style="width:20%; padding:12px; border:1px solid #ddd;">Reference Images</th>
      <th style="width:20%; padding:12px; border:1px solid #ddd;">Reference Audios</th>
      <th style="width:30%; padding:12px; border:1px solid #ddd;">Text prompts</th>
      <th style="width:30%; padding:12px; border:1px solid #ddd;">Generated Videos (GIF)</th>
    </tr>
  </thead>
  <tbody>

    <tr>
      <td align="center" style="vertical-align:middle; padding:12px; border:1px solid #ddd;">
        <img src="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/images/ref_68.png" alt="Reference Image 1" style="max-height:200px; max-width:100%;">
      </td>
      <td align="center" style="vertical-align:middle; padding:12px; border:1px solid #ddd;">
        <audio controls src="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/audio/00068_audio.mp3">
        <br>
        <a href="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/audio/00068_audio.mp3" style="font-size:12px; color:#666;">⤓ Download</a>
      </td>
      <td style="line-height:1.8; vertical-align:middle; padding:12px; border:1px solid #ddd;">
        A man stands at the podium in OpenAI's luxurious conference room, behind him a massive electronic screen displays the company's glowing profit data. He grips the microphone firmly, gazes across the audience below, and announces in a steady tone: &lt;S&gt;The board wants to sell OpenAI to Zuckerberg, which is unacceptable.&lt;E&gt;
      </td>
      <td align="center" style="vertical-align:middle; padding:12px; border:1px solid #ddd;">
        <img src="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/videos/demo/68.gif" alt="Generated Video 1" style="width:100%; max-width:600px; height:auto; min-height:200px;">
      </td>
    </tr>

    <tr>
      <td align="center" style="vertical-align:middle; padding:12px; border:1px solid #ddd;">
        <img src="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/images/ref_69.png" alt="Reference Image 2" style="max-height:200px; max-width:100%;">
      </td>
      <td align="center" style="vertical-align:middle; padding:12px; border:1px solid #ddd;">
        <audio controls src="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/audio/00069_audio.wav">
        <br>
        <a href="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/audio/00069_audio.wav" style="font-size:12px; color:#666;">⤓ Download</a>
      </td>
      <td style="line-height:1.8; vertical-align:middle; padding:12px; border:1px solid #ddd;">
        A woman stands before the iconic Rockefeller Center Christmas Tree, its thousands of lights reflecting in her eyes as snow begins to fall gently around her. Wearing a tartan scarf and holding a cup of steaming cocoa, she brings her mittened hands together and speaks softly into the frosty air: &lt;S&gt;May the spirit of Christmas fill your heart throughout the coming year.&lt;E&gt;
      </td>
      <td align="center" style="vertical-align:middle; padding:12px; border:1px solid #ddd;">
        <img src="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/videos/demo/69.gif" alt="Generated Video 2" style="width:100%; max-width:600px; height:auto; min-height:200px;">
      </td>
    </tr>

    <tr>
      <td align="center" style="vertical-align:middle; padding:12px; border:1px solid #ddd;">
        <img src="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/images/ref_70.png" alt="Reference Image 3" style="max-height:200px; max-width:100%;">
      </td>
      <td align="center" style="vertical-align:middle; padding:12px; border:1px solid #ddd;">
        <audio controls src="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/audio/00070_audio.wav">
        <br>
        <a href="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/audio/00070_audio.wav" style="font-size:12px; color:#666;">⤓ Download</a>
      </td>
      <td style="line-height:1.8; vertical-align:middle; padding:12px; border:1px solid #ddd;">
        A man stands on a bustling street in Shanghai, the air thick with the festive atmosphere of Chinese Lunar New Year, with numerous red lanterns hanging in clusters overhead. He blends seamlessly into the vibrant surroundings, then clasps his hands together in a traditional gesture of greeting and says warmly: &lt;S&gt;Wishing everyone a Happy New Year and joy every single day.&lt;E&gt;
      </td>
      <td align="center" style="vertical-align:middle; padding:12px; border:1px solid #ddd;">
        <img src="https://raw.githubusercontent.com/OmniCustom-project/OmniCustom/main/assets/videos/demo/70.gif" alt="Generated Video 3" style="width:100%; max-width:600px; height:auto; min-height:200px;">
      </td>
    </tr>

  </tbody>
</table>


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
