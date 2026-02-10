# OmniCustom (Inference Only)

This repository has been trimmed to an **inference-only** version. Training code is not included.

## 1. Overview

OmniCustom is used for multimodal video generation inference.  
Main entry:

- `./new_infer.py`

Supported modes:

- `t2v`: text-to-video
- `i2v`: image-to-video
- `id2v`: identity-conditioned video (IP image / IP audio)
- `t2i2v`: text-to-image-to-video

## 2. Current Structure (After Cleanup)

- `new_infer.py`: main inference entry
- `ovi_fusion_engine.py`: inference engine
- `configs/inference/inference_fusion.yaml`: inference config
- `modules/`: model and codec modules
- `utils/`: I/O, preprocessing, scheduler utilities
- `distributed_comms/`: distributed/SP communication
- `example_prompts/`: sample prompts and assets
- `inference.sh`: single-GPU launch example
- `setup_uv.sh`: one-click environment setup with `uv`

## 3. Environment Setup (uv, One-Click)

Recommended: Python `3.10` with CUDA.

One-click setup:

```bash
cd .
bash ./setup_uv.sh
```

What it does:

- checks/installs `uv` (if missing)
- creates `.venv` with Python `3.10` (or `$PYTHON_VERSION`)
- installs dependencies from `requirements.txt`

Optional: choose another Python version:

```bash
PYTHON_VERSION=3.11 bash ./setup_uv.sh
```

## 4. Checkpoint and Path Requirements

Make sure the paths in `./configs/inference/inference_fusion.yaml` are valid, especially:

- `ckpt_dir`
- `ckpt_name`
- `lora_path`
- `face_embedder_ckpt_dir`
- `audio_embedder_ckpt_dir`

Based on current code, `ckpt_dir` should contain at least:

- `Wan2.2-TI2V-5B/Wan2.2_VAE.pth`
- `Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth`
- `Wan2.2-TI2V-5B/google/umt5-xxl` (tokenizer directory)
- `MMAudio/ext_weights/v1-16.pth`
- `MMAudio/ext_weights/best_netG.pt`
- your fusion checkpoint file (`ckpt_name`)

## 5. Quick Start

Single GPU:

```bash
cd .
CUDA_VISIBLE_DEVICES=0 .venv/bin/python new_infer.py --config-file ./configs/inference/inference_fusion.yaml
```

Or run:

```bash
bash ./inference.sh
```

Multi-GPU example:

```bash
cd .
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 new_infer.py --config-file ./configs/inference/inference_fusion.yaml
```

Notes:

- `sp_size` must satisfy `world_size % sp_size == 0`
- In multi-GPU mode, samples are sharded by SP group

## 6. Key Config Fields

In `./configs/inference/inference_fusion.yaml`:

- `mode`: `id2v / t2v / i2v / t2i2v`
- `text_prompt`: plain text, or path to `.csv/.tsv`
- `image_path`: first-frame image for `i2v`
- `ip_image_path` / `ip_audio_path`: conditioning inputs for `id2v`
- `video_frame_height_width`: output resolution (for example `[560, 992]`)
- `sample_steps`, `solver_name`, `shift`: sampling parameters
- `video_guidance_scale`, `audio_guidance_scale`: guidance scales
- `crop_face`: enable face crop
- `each_example_n_times`: number of repeats per input
- `seed`: base random seed (incremented per repeat)
- `output_dir`: output directory

## 7. Prompt Input Format

`text_prompt` supports:

1. Direct text
2. A `.csv/.tsv` file (must contain `text_prompt` column)

Optional columns by mode:

- `i2v`: `image_path`
- `id2v`: `ip_image_path`, `ip_audio_path`

Example CSV:

```csv
text_prompt,image_path,ip_image_path,ip_audio_path
"a woman speaks to camera",./example_prompts/pngs/0.png,./example_prompts/pngs/girl-001.jpg,./example_prompts/audios/girl-001.mp3
```

## 8. Output Layout

Outputs are grouped by condition folders:

- `ip_image_True_ip_audio_True/`
- `ip_image_True_ip_audio_False/`
- `ip_image_False_ip_audio_True/`
- `ip_image_False_ip_audio_False/`

Filename format:

`{5-digit-seq}_crop-{crop_face}_{prompt_short}_{HxW}_{seed}_{rank}.mp4`

If a generated first-frame image exists, a same-name `.png` is also saved.

## 9. Troubleshooting

- Path not found: check checkpoint paths and input file paths first.
- No output on multi-GPU: verify `CUDA_VISIBLE_DEVICES`, `torchrun` args, and `sp_size`.
- `i2v` failure: make sure `image_path` (or CSV `image_path`) exists and is readable.

## 10. Notes

- This repo currently does not include training scripts.
- Use `new_infer.py` as the default inference entry.
