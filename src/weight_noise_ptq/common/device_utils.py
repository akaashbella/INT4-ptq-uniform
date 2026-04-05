"""Centralized ``torch.device`` resolution (explicit CUDA errors, no silent CPU fallback)."""

from __future__ import annotations

import torch


def resolve_torch_device(device: str | torch.device | None) -> torch.device:
    """Resolve a device for training/eval.

    - ``None`` → ``cuda`` if available, else ``cpu``.
    - If ``cuda`` is requested (string or :class:`~torch.device`) but unavailable,
      raises with a clear message (avoids subtle CPU runs labeled as CUDA in logs).
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.device(device) if isinstance(device, str) else device
    if d.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA was requested (device={d!r}) but torch.cuda.is_available() is False. "
                "Use --device cpu or run on a machine with a visible CUDA device."
            )
        if d.index is not None and int(d.index) >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {d.index} is out of range "
                f"(torch.cuda.device_count() == {torch.cuda.device_count()})."
            )
    return d
