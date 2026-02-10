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

<table style="width: 100%; border-collapse: collapse; text-align: center; border: 1px solid #ccc; margin: 10px 0;"> <!-- 表头行 --> <tr style="background-color: #f5f5f5;"> <th style="padding: 12px; border: 1px solid #ccc;"><strong>Reference Images</strong></th> <th style="padding: 12px; border: 1px solid #ccc;"><strong>Reference Audios</strong></th> <th style="padding: 12px; border: 1px solid #ccc;"><strong>Text prompts</strong></th> <th style="padding: 12px; border: 1px solid #ccc;"><strong>Generated Videos (480P)</strong></th> </tr>

<!-- 示例1（带换行） --> <tr> <td style="padding: 10px; border: 1px solid #ccc; vertical-align: middle;"> <img src="assets/images/ref_68.png" alt="Reference Image 1" style="height: 150px;"> </td> <td style="padding: 10px; border: 1px solid #ccc; vertical-align: middle;"> <audio src="assets/audio/00068_audio.wav" controls style="width: 200px;"> Your browser does not support the audio tag. </audio> </td> <td style="padding: 10px; border: 1px solid #ccc; vertical-align: middle; line-height: 1.8;"> A man stands at the podium in OpenAI's luxurious <br>conference room, behind him a massive electronic  <br> screen displays the company's glowing  profit data. <br> He grips the microphone firmly, gazes across the  <br> audience below, and announces in a steady tone:  <br> &lt;S&gt;The board wants to sell OpenAI to Zuckerberg,  <br> which  is unacceptable.&lt;E&gt; </td> <td style="padding: 10px; border: 1px solid #ccc; vertical-align: middle;"> <img src="assets/videos/68.mp4" alt="Generated Video 1" style="width: 350px;"> </td> </tr>

<!-- 示例2（带换行） --> <tr> <td style="padding: 10px; border: 1px solid #ccc; vertical-align: middle;"> <img src="assets/images/ref_69.png" alt="Reference Image 2" style="height: 150px;"> </td> <td style="padding: 10px; border: 1px solid #ccc; vertical-align: middle;"> <audio src="assets/audio/00069_audio.wav" controls style="width: 200px;"> Your browser does not support the audio tag. </audio> </td> <td style="padding: 10px; border: 1px solid #ccc; vertical-align: middle; line-height: 1.8;"> A woman stands before the iconic Rockefeller  <br> Center Christmas Tree, its thousands of lights  <br> reflecting in  her eyes as snow begins to fall gently  <br>around her. Wearing a tartan scarf and holding  <br> a cup of steaming cocoa, she brings her mittened <br> hands together and speaks softly into the frosty  <br> air: &lt;S&gt;May the spirit of Christmas fill your heart  <br> throughout the coming year.&lt;E&gt;</td> <td style="padding: 10px; border: 1px solid #ccc; vertical-align: middle;"> <img src="assets/videos/69.mp4" alt="Generated Video 2" style="width: 350px;"> </td> </tr>

<!-- 示例3（带换行） --> <tr> <td style="padding: 10px; border: 1px solid #ccc; vertical-align: middle;"> <img src="assets/images/ref_70.png" alt="Reference Image 3" style="height: 150px;"> </td> <td style="padding: 10px; border: 1px solid #ccc; vertical-align: middle;"> <audio src="assets/audio/00070_audio.wav" controls style="width: 200px;"> Your browser does not support the audio tag. </audio> </td> <td style="padding: 10px; border: 1px solid #ccc; vertical-align: middle; line-height: 1.8;"> A man stands on a bustling street in Shanghai, <br> the air thick with the festive atmosphere of <br> Chinese Lunar New Year, with numerous red <br> lanterns hanging in clusters overhead. He blends <br> seamlessly into the vibrant surroundings, <br> then clasps his hands together in a traditional <br> gesture of greeting and says warmly: &lt;S&gt;Wishing<br> everyone a Happy New Year and joy every single <br> day.&lt;E&gt;</td> <td style="padding: 10px; border: 1px solid #ccc; vertical-align: middle;"> <img src="assets/videos/70.mp4" alt="Generated Video 3" style="width: 350px;"> </td> </tr>

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
