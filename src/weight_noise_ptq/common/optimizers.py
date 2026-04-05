"""Optimizer factories from :class:`OptimConfig`."""

from __future__ import annotations

import torch
from torch import nn

from weight_noise_ptq.common.config import OptimConfig


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    """Build SGD or Adam from config (locked experiment uses SGD for classification, Adam for compression)."""
    name = cfg.name.lower()
    params = model.parameters()
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(cfg.lr),
            momentum=float(cfg.momentum),
            weight_decay=float(cfg.weight_decay),
        )
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )
    raise ValueError(f"Unsupported optimizer {cfg.name!r}; use 'sgd' or 'adam'.")
