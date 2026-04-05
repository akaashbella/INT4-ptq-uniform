"""Deterministic seeding for Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, *, deterministic_cuda: bool = True) -> None:
    """Set global RNG seeds and optional cuDNN deterministic flags.

    Full bitwise reproducibility across CUDA ops is not guaranteed by PyTorch;
    this sets the usual flags to reduce nondeterminism where implemented.
    """
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)

    if deterministic_cuda and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def worker_init_fn(worker_id: int, base_seed: int) -> None:
    """``DataLoader`` worker initializer for reproducible augmentation."""
    ss = int(base_seed) + int(worker_id)
    np.random.seed(ss)
    random.seed(ss)
    torch.manual_seed(ss)
