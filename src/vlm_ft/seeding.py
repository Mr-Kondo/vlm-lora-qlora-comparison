"""Seed handling shared by every run so LoRA and QLoRA see the same data order."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) from a single value."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass
