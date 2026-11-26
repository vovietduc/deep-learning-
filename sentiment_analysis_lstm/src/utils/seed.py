"""Utilities for reproducible Python, NumPy and PyTorch experiments."""
from __future__ import annotations

import os
import random


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for common libraries.

    PyTorch and NumPy are imported optionally so this helper can still be
    imported in lightweight environments. When installed, their RNGs are set.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            # Warn rather than fail for operations without deterministic kernels.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # Compatibility with older PyTorch versions.
                torch.use_deterministic_algorithms(True)
    except ImportError:
        pass
