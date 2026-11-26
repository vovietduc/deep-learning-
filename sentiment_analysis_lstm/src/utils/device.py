"""Device selection helpers for CPU/CUDA execution."""
from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA when requested and available; otherwise return CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info(device: torch.device | None = None) -> dict[str, str | bool]:
    """Return a small, serializable summary of the selected device."""
    selected = device or get_device()
    info: dict[str, str | bool] = {
        "device": str(selected),
        "cuda_available": torch.cuda.is_available(),
        "is_cuda": selected.type == "cuda",
    }
    if selected.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(selected)
    else:
        info["device_name"] = "CPU"
    return info
