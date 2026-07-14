"""
Project-wide reproducibility and device helpers.
"""
from __future__ import annotations

import random
from typing import Any, Mapping

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, PyTorch, and every visible CUDA device.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def default_device_name() -> str:
    """
    Choose CUDA, then Apple MPS, then CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def training_dtype(device: torch.device, bf16: bool) -> torch.dtype:
    """
    Select a training dtype suitable for the requested device.
    """
    if device.type == "cpu":
        return torch.float32
    return torch.bfloat16 if bf16 else torch.float16


def move_tensors_to_device(
    inputs: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """
    Move tensor dictionary values while preserving metadata values.
    """
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
