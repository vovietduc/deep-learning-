"""Compute-device selection helpers."""
from __future__ import annotations

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return CUDA when available and requested; otherwise return CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device | None = None) -> str:
    """Return a short human-readable description of the selected device."""
    device = device or get_device()
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"CUDA: {torch.cuda.get_device_name(index)}"
    return "CPU"
