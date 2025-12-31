from __future__ import annotations

from pathlib import Path
import time
from typing import Optional, List, Dict, Any

try:
    from .paths import get_outputs_dir
    from .logger import get_logger
    from . import db
except ImportError:  # pragma: no cover
    from paths import get_outputs_dir  # type: ignore
    from logger import get_logger  # type: ignore
    import db  # type: ignore

logger = get_logger("text2video.storage")

def sanitize_prompt(prompt: str, max_len: int = 30) -> str:
    safe = "".join(c for c in prompt[:max_len] if c.isalnum() or c in "-_")
    return safe or "image"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def record_generation(
    prompt: str,
    steps: int,
    width: int,
    height: int,
    filename: str,
    generation_time: float,
    file_size_kb: float,
    model: str,
    precision: str,
    seed: Optional[int],
    cfg_scale: float = 0.0,
    loras: Optional[List[Dict[str, Any]]] = None,
):
    """
    Persist a generation record to the DB. Best-effort with logging.
    Returns the new record ID or None on failure.
    """
    try:
        return db.add_generation(
            prompt=prompt,
            steps=steps,
            width=width,
            height=height,
            filename=filename,
            generation_time=generation_time,
            file_size_kb=file_size_kb,
            model=model,
            cfg_scale=cfg_scale,
            seed=seed,
            status="succeeded",
            precision=precision,
            loras=loras,
        )
    except Exception as e:  # pragma: no cover
        logger.error(f"Failed to record generation to DB: {e}")
        return None
