from __future__ import annotations

import os
import sys
import warnings
import traceback
import argparse
# Import from paths module for config access
try:
    from .paths import load_config
except ImportError:
    from paths import load_config

# Silence the noisy CUDA autocast warning on Mac
warnings.filterwarnings(
    "ignore",
    message="User provided device_type of 'cuda', but CUDA is not available",
    category=UserWarning,
)
import time
import torch
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from safetensors.torch import load_file
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from dfloat11 import DFloat11Model
# Import from hardware module
try:
    from .hardware import (
        PrecisionId,
        MODEL_ID_MAP,
        get_available_models,
        should_enable_attention_slicing,
        get_ram_gb,
        detect_device,
    )
    from .logger import get_logger
except ImportError:
    from hardware import (
        PrecisionId,
        MODEL_ID_MAP,
        get_available_models,
        should_enable_attention_slicing,
        get_ram_gb,
        detect_device,
    )
    from logger import get_logger

logger = get_logger("text2video.engine")


def log_info(message: str):
    logger.info(message)

def log_warn(message: str):
    logger.warning(message)


warnings.filterwarnings(
    "ignore",
    message="`torch_dtype` is deprecated! Use `dtype` instead!",
    category=FutureWarning,
)

_cached_pipe = None

def load_pipeline(device: str = None) -> WanPipeline:

    global _cached_pipe

    if _cached_pipe is not None:
        log_info(f"using model exists")
        return _cached_pipe

    if device is None:
        device = detect_device()
    log_info(f"using device: {device}")



    log_info(f"using model: DFloat11")

    vae = AutoencoderKLWan.from_pretrained("/home/model/Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="vae",
                                           torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained("/home/model/Wan-AI/Wan2.2-T2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16,
                                       local_files_only=True)

    # Load DFloat11 models
    DFloat11Model.from_pretrained(
        "/home/model/DFloat11/Wan2.2-T2V-A14B-DF11",
        device="cpu",
        cpu_offload=True,
        bfloat16_model=pipe.transformer,
        local_files_only=True,
    )
    DFloat11Model.from_pretrained(
        "/home/model/DFloat11/Wan2.2-T2V-A14B-2-DF11",
        device="cpu",
        cpu_offload=True,
        bfloat16_model=pipe.transformer_2,
        local_files_only=True,
    )

    pipe.enable_model_cpu_offload()

    log_info(f"model loading first time")

    _cached_pipe = pipe
    return pipe

def generate_video(
    prompt: str,
):
    try:
        start_time = time.time()

        pipe = load_pipeline()

        log_info(f"generating video for prompt: {prompt!r}")

        with torch.inference_mode():
            video = pipe(
                prompt=prompt,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                height=1280,
                width=720,
                num_frames=150,
                guidance_scale=4.0,
                guidance_scale_2=3.0,
                num_inference_steps=30,
            ).frames[0]
    except Exception as e:
        logger.error(f"Error generating video: {e}")
    finally:
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        # Clear CUDA cache on error to free GPU memory for next request
        if torch.cuda.is_available():
            log_warn("Clearing CUDA cache after error")
            torch.cuda.empty_cache()

    return video

def cleanup_memory():
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
