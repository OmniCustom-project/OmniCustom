import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from distributed_comms.parallel_states import (
    get_sequence_parallel_state,
    initialize_sequence_parallel_state,
    nccl_info,
)
from distributed_comms.util import get_global_rank, get_local_rank, get_world_size
from ovi_fusion_engine import OviFusionEngine
from utils.io_utils import save_video
from utils.processing_utils import (
    format_prompt_for_filename,
    validate_and_process_user_prompt,
)
from utils.utils import get_arguments

GenerationItem = Tuple[str, Optional[str], Optional[str], Optional[str]]
ALLOWED_MODES = {"id2v", "t2v", "i2v", "t2i2v"}


@dataclass(frozen=True)
class RuntimeState:
    world_size: int
    global_rank: int
    local_rank: int
    device: int
    sp_rank: int
    sp_group_id: int
    num_sp_groups: int


class SequenceNumberManager:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self._next_by_condition: Dict[str, int] = {}

    def next(self, condition_dir: str) -> int:
        if condition_dir not in self._next_by_condition:
            self._next_by_condition[condition_dir] = self._scan_next(condition_dir)
        value = self._next_by_condition[condition_dir]
        self._next_by_condition[condition_dir] += 1
        return value

    def _scan_next(self, condition_dir: str) -> int:
        condition_output_dir = os.path.join(self.output_dir, condition_dir)
        if not os.path.exists(condition_output_dir):
            return 1

        max_sequence = 0
        for filename in os.listdir(condition_output_dir):
            if not (filename.endswith(".mp4") or filename.endswith(".png")):
                continue
            parts = filename.split("_")
            if parts and parts[0].isdigit():
                max_sequence = max(max_sequence, int(parts[0]))
        return max_sequence + 1


def _init_logging(rank: int) -> None:
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def _initialize_runtime(config, args) -> RuntimeState:
    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()
    device = local_rank
    torch.cuda.set_device(local_rank)

    sp_size = config.get("sp_size", 1)
    assert sp_size <= world_size and world_size % sp_size == 0, (
        "sp_size must be less than or equal to world_size and world_size "
        "must be divisible by sp_size."
    )

    _init_logging(global_rank)

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=global_rank,
            world_size=world_size,
        )
    else:
        assert (
            sp_size == 1
        ), f"When world_size is 1, sp_size must also be 1, but got {sp_size}."

    initialize_sequence_parallel_state(sp_size)
    logging.info("Using SP: %s, SP_SIZE: %s", get_sequence_parallel_state(), sp_size)

    args.local_rank = local_rank
    args.device = device

    if get_sequence_parallel_state():
        runtime_sp_size = nccl_info.sp_size
        sp_rank = nccl_info.rank_within_group
        sp_group_id = global_rank // runtime_sp_size
        num_sp_groups = world_size // runtime_sp_size
    else:
        sp_rank = 0
        sp_group_id = global_rank
        num_sp_groups = world_size

    return RuntimeState(
        world_size=world_size,
        global_rank=global_rank,
        local_rank=local_rank,
        device=device,
        sp_rank=sp_rank,
        sp_group_id=sp_group_id,
        num_sp_groups=num_sp_groups,
    )


def _prepare_eval_data(config) -> List[GenerationItem]:
    mode = config.get("mode")
    assert mode in ALLOWED_MODES, (
        f"Invalid mode {mode}, must be one of {sorted(ALLOWED_MODES)}"
    )

    text_prompt = config.get("text_prompt")
    image_path = config.get("image_path")
    ip_image_path = config.get("ip_image_path")
    ip_audio_path = config.get("ip_audio_path")

    text_prompts, image_paths, ip_image_paths, ip_audio_paths = (
        validate_and_process_user_prompt(
            text_prompt,
            image_path,
            ip_image_path,
            ip_audio_path,
            mode=mode,
        )
    )

    if mode != "i2v":
        logging.info(
            "mode: %s, setting all image_paths, ip_image_paths and "
            "ip_audio_paths to None",
            mode,
        )
        image_paths = [None] * len(text_prompts)
    else:
        assert all(
            p is not None and os.path.isfile(p) for p in image_paths
        ), f"In i2v mode, all image paths must be provided.{image_paths}"

    return list(zip(text_prompts, image_paths, ip_image_paths, ip_audio_paths))


def _split_eval_data_for_current_rank(
    all_eval_data: Sequence[GenerationItem],
    runtime: RuntimeState,
    require_sample_padding: bool = False,
) -> List[GenerationItem]:
    total_files = len(all_eval_data)
    if total_files == 0:
        logging.error("ERROR: No evaluation files found")
        return []

    eval_data = list(all_eval_data)
    remainder = total_files % runtime.num_sp_groups
    if require_sample_padding and remainder != 0:
        pad_count = runtime.num_sp_groups - remainder
        eval_data += [eval_data[0]] * pad_count

    return eval_data[runtime.sp_group_id :: runtime.num_sp_groups]


def _validate_optional_path(path: Optional[str], display_name: str) -> Optional[str]:
    if path is None:
        return None
    if not os.path.isfile(path):
        logging.warning("%s %s not exists, using `None` instead", display_name, path)
        return None
    return path


def _frame_size_string(video_frame_height_width: Optional[Sequence[int]]) -> str:
    if video_frame_height_width is None:
        raise ValueError("video_frame_height_width must be provided in config.")
    return "x".join(map(str, video_frame_height_width))


def _build_output_path(
    output_dir: str,
    sequence_manager: SequenceNumberManager,
    text_prompt: str,
    ip_image_path: Optional[str],
    ip_audio_path: Optional[str],
    crop_face: bool,
    video_frame_height_width: Optional[Sequence[int]],
    seed: int,
    global_rank: int,
) -> str:
    condition_dir = (
        f"ip_image_{ip_image_path is not None}_ip_audio_{ip_audio_path is not None}"
    )
    condition_output_dir = os.path.join(output_dir, condition_dir)
    os.makedirs(condition_output_dir, exist_ok=True)

    sequence_number = sequence_manager.next(condition_dir)
    sequence_str = f"{sequence_number:05d}"
    formatted_prompt = format_prompt_for_filename(text_prompt)
    frame_size = _frame_size_string(video_frame_height_width)
    output_filename = (
        f"{sequence_str}_crop-{crop_face}_{formatted_prompt}_"
        f"{frame_size}_{seed}_{global_rank}.mp4"
    )
    return os.path.join(condition_output_dir, output_filename)


def main(config, args) -> None:
    runtime = _initialize_runtime(config, args)
    target_dtype = torch.bfloat16

    all_eval_data = _prepare_eval_data(config)
    this_rank_eval_data = _split_eval_data_for_current_rank(all_eval_data, runtime)

    self_lora = config.get("self_lora", True)
    logging.info("Loading OVI Fusion Engine...")
    ovi_engine = OviFusionEngine(
        config=config,
        device=runtime.device,
        target_dtype=target_dtype,
        self_lora=self_lora,
    )
    logging.info("OVI Fusion Engine loaded!")

    output_dir = config.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    sequence_manager = SequenceNumberManager(output_dir)

    generation_kwargs = {
        "video_frame_height_width": config.get("video_frame_height_width"),
        "seed": config.get("seed", 100),
        "solver_name": config.get("solver_name", "unipc"),
        "sample_steps": config.get("sample_steps", 50),
        "shift": config.get("shift", 5.0),
        "video_guidance_scale": config.get("video_guidance_scale", 4.0),
        "audio_guidance_scale": config.get("audio_guidance_scale", 3.0),
        "slg_layer": config.get("slg_layer", 11),
        "video_negative_prompt": config.get("video_negative_prompt", ""),
        "audio_negative_prompt": config.get("audio_negative_prompt", ""),
    }
    crop_face = config.get("crop_face", False)
    each_example_n_times = config.get("each_example_n_times", 1)

    for text_prompt, image_path, ip_image_path, ip_audio_path in tqdm(this_rank_eval_data):
        ip_image_path = _validate_optional_path(ip_image_path, "IP Image")
        ip_audio_path = _validate_optional_path(ip_audio_path, "IP Audio")

        for idx in range(each_example_n_times):
            current_seed = generation_kwargs["seed"] + idx
            generated_video, generated_audio, generated_image = ovi_engine.generate(
                text_prompt=text_prompt,
                image_path=image_path,
                ip_image_path=ip_image_path,
                ip_audio_path=ip_audio_path,
                video_frame_height_width=generation_kwargs["video_frame_height_width"],
                seed=current_seed,
                solver_name=generation_kwargs["solver_name"],
                sample_steps=generation_kwargs["sample_steps"],
                shift=generation_kwargs["shift"],
                video_guidance_scale=generation_kwargs["video_guidance_scale"],
                audio_guidance_scale=generation_kwargs["audio_guidance_scale"],
                slg_layer=generation_kwargs["slg_layer"],
                video_negative_prompt=generation_kwargs["video_negative_prompt"],
                audio_negative_prompt=generation_kwargs["audio_negative_prompt"],
            )

            if runtime.sp_rank != 0:
                continue

            output_path = _build_output_path(
                output_dir=output_dir,
                sequence_manager=sequence_manager,
                text_prompt=text_prompt,
                ip_image_path=ip_image_path,
                ip_audio_path=ip_audio_path,
                crop_face=crop_face,
                video_frame_height_width=generation_kwargs["video_frame_height_width"],
                seed=current_seed,
                global_rank=runtime.global_rank,
            )
            save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
            logging.info("Saved video to: %s", output_path)

            if generated_image is not None:
                image_output_path = output_path.replace(".mp4", ".png")
                generated_image.save(image_output_path)
                logging.info("Saved image to: %s", image_output_path)


if __name__ == "__main__":
    args = get_arguments()
    config = OmegaConf.load(args.config_file)
    main(config=config, args=args)
