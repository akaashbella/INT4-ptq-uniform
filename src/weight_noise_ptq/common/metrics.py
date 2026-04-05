"""Classification and compression metric helpers (PSNR, bpp, RD-style scalars)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def top1_accuracy(logits: Tensor, targets: Tensor) -> float:
    """Fraction of correct predictions (top-1)."""
    pred = logits.argmax(dim=1)
    correct = (pred == targets).float().mean().item()
    return float(correct)


def cross_entropy_mean(logits: Tensor, targets: Tensor) -> float:
    """Mean cross-entropy loss for a batch."""
    return float(torch.nn.functional.cross_entropy(logits, targets).item())


@dataclass
class ClassificationAggregate:
    """Running sums for loss and top-1 over a pass.

    ``loss`` is assumed to be ``CrossEntropyLoss(..., reduction='mean')``, i.e. mean
    over the batch. Global mean CE is ``sum_b (loss_b * n_b) / sum_b n_b``.
    """

    sum_loss_times_batch: float = 0.0
    sum_top1: float = 0.0
    n_batches: int = 0
    n_samples: int = 0

    def update(self, logits: Tensor, targets: Tensor, loss: Tensor | float) -> None:
        bs = int(targets.shape[0])
        lm = float(loss) if not isinstance(loss, Tensor) else float(loss.item())
        self.sum_loss_times_batch += lm * float(bs)
        self.sum_top1 += top1_accuracy(logits, targets) * bs
        self.n_batches += 1
        self.n_samples += bs

    def mean_loss(self) -> float:
        if self.n_samples == 0:
            return 0.0
        return self.sum_loss_times_batch / float(self.n_samples)

    def mean_top1(self) -> float:
        if self.n_samples == 0:
            return 0.0
        return self.sum_top1 / self.n_samples


def drop_from_fp32(metric: float, fp32_ref: float) -> float:
    """``metric - fp32_ref`` (e.g. lower accuracy is more negative drop)."""
    return float(metric) - float(fp32_ref)


def retention_ratio(metric: float, fp32_ref: float) -> float:
    """``metric / fp32_ref`` when ``fp32_ref != 0``; else ``float('nan')``."""
    ref = float(fp32_ref)
    if ref == 0.0:
        return float("nan")
    return float(metric) / ref


def psnr_from_mse(mse: Tensor | float, *, max_pixel: float = 1.0) -> float:
    """PSNR in dB for MSE on images scaled to ``[0, max_pixel]``."""
    m = float(mse) if not isinstance(mse, Tensor) else float(mse.item())
    if m <= 0.0:
        return float("inf")
    mx = float(max_pixel)
    return float(10.0 * math.log10((mx * mx) / m))


def estimate_bpp_from_likelihoods(
    likelihoods: dict[str, Tensor],
    num_image_pixels: int | Tensor,
) -> Tensor:
    """Bits per pixel from CompressAI-style ``likelihoods`` dict.

    ``num_image_pixels`` is ``N * H * W`` (batch × spatial pixels), matching
    common CompressAI examples (total bits / spatial pixels).
    """
    if isinstance(num_image_pixels, Tensor):
        n_pix = num_image_pixels.to(dtype=torch.float64)
    else:
        n_pix = torch.tensor(float(num_image_pixels), dtype=torch.float64)
    total = torch.zeros((), dtype=torch.float64)
    for v in likelihoods.values():
        # clamp for numerical stability
        lik = v.clamp(min=1e-10)
        bits = -torch.log2(lik)
        total = total + bits.sum()
    return (total / n_pix).to(dtype=torch.float32)


def rate_distortion_loss(
    mse: Tensor,
    bpp: Tensor,
    lambda_rd: float,
) -> Tensor:
    """``mse + lambda_r_d * bpp`` (scalar tensor)."""
    return mse + float(lambda_rd) * bpp


def psnr_drop_from_fp32(psnr: float, fp32_psnr: float) -> float:
    """``psnr - fp32_psnr``."""
    return float(psnr) - float(fp32_psnr)


def bpp_shift_from_fp32(bpp: float, fp32_bpp: float) -> float:
    """``bpp - fp32_bpp``."""
    return float(bpp) - float(fp32_bpp)


def flatten_metrics_dict(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten nested metric dicts for logging (one level)."""
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_metrics_dict(v, prefix=key + "."))
        elif isinstance(v, (float, int)):
            out[key] = float(v)
    return out
