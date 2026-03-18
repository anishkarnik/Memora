"""Hardware detection and adaptive configuration.

Detects system capabilities once at import time and exposes adaptive constants
that scale Memora's resource usage to the available hardware.

Profile tiers:
  "lite"        — RAM <= 8GB AND no discrete GPU
  "standard"    — RAM > 8GB OR GPU with < 6GB VRAM
  "performance" — RAM >= 16GB AND GPU with >= 6GB VRAM
"""
import json
import os
from pathlib import Path
from typing import Optional

import psutil

# ── Detection (cached at module level) ────────────────────────────────────────

CPU_CORES: int = os.cpu_count() or 1
RAM_BYTES: int = psutil.virtual_memory().total
RAM_GB: float = round(RAM_BYTES / (1024 ** 3), 1)

GPU_AVAILABLE: bool = False
GPU_NAME: Optional[str] = None
GPU_VRAM_GB: float = 0.0

try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_VRAM_GB = round(torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 1)
except Exception:
    pass

# ── Settings path ─────────────────────────────────────────────────────────────

_SETTINGS_PATH = Path(os.environ.get("MEMORA_DATA_DIR", str(Path.home() / ".memora"))) / "settings.json"


def _read_setting(key: str, default=None):
    try:
        if _SETTINGS_PATH.exists():
            return json.loads(_SETTINGS_PATH.read_text()).get(key, default)
    except Exception:
        pass
    return default


# ── Profile detection ─────────────────────────────────────────────────────────

def _detect_profile() -> str:
    if RAM_GB >= 16 and GPU_AVAILABLE and GPU_VRAM_GB >= 6:
        return "performance"
    if RAM_GB > 8 or (GPU_AVAILABLE and GPU_VRAM_GB > 0):
        return "standard"
    return "lite"


def get_profile() -> str:
    """Return the active performance profile (auto-detected or user override)."""
    override = _read_setting("performance_profile", "auto")
    if override in ("lite", "standard", "performance"):
        return override
    return _detect_profile()


# ── Adaptive constants ────────────────────────────────────────────────────────

def get_max_workers() -> int:
    return {"lite": 1, "standard": 2, "performance": 4}[get_profile()]


def get_face_det_size() -> tuple[int, int]:
    if get_profile() == "lite":
        return (320, 320)
    return (640, 640)


def get_clip_batch_size() -> int:
    return {"lite": 1, "standard": 4, "performance": 16}[get_profile()]


def get_caption_batch_size() -> int:
    return {"lite": 1, "standard": 2, "performance": 8}[get_profile()]


def get_faiss_save_interval() -> int:
    return {"lite": 20, "standard": 50, "performance": 100}[get_profile()]


def get_db_commit_interval() -> int:
    return {"lite": 10, "standard": 25, "performance": 50}[get_profile()]


def should_use_float16() -> bool:
    return GPU_AVAILABLE


def skip_captioning() -> bool:
    return bool(_read_setting("skip_captioning", False))


def skip_face_detection() -> bool:
    return bool(_read_setting("skip_face_detection", False))


# ── Info endpoint helper ──────────────────────────────────────────────────────

def get_hardware_info() -> dict:
    return {
        "cpu_cores": CPU_CORES,
        "ram_gb": RAM_GB,
        "gpu_name": GPU_NAME,
        "gpu_vram_gb": GPU_VRAM_GB,
        "has_cuda": GPU_AVAILABLE,
        "profile": get_profile(),
        "detected_profile": _detect_profile(),
    }
