"""Optimizer factories from :class:`OptimConfig`."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from weight_noise_ptq.common.config import OptimConfig


def build_optimizer(
    model_or_params: nn.Module | Iterable[nn.Parameter],
    cfg: OptimConfig,
) -> torch.optim.Optimizer:
    """Build SGD or Adam over a module or an explicit parameter list.

    Compression training passes **main-network** parameters only; auxiliary
    entropy parameters use a second optimizer built the same way.
    """
    name = cfg.name.lower()
    if isinstance(model_or_params, nn.Module):
        params = list(model_or_params.parameters())
    else:
        params = list(model_or_params)
    if not params:
        raise ValueError("build_optimizer: empty parameter list")
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
