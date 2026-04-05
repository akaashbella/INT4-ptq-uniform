"""Shared evaluation logic for :mod:`scripts.eval_fp32` and :mod:`scripts.eval_quant`.

Keeps metric definitions and JSON payload shapes aligned across eval entrypoints.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from weight_noise_ptq.classification.registry import build_classification_model
from weight_noise_ptq.common.metrics import (
    ClassificationAggregate,
    estimate_bpp_from_likelihoods,
    psnr_from_mse,
    rate_distortion_loss,
)
from weight_noise_ptq.common.quantization import quantize_eligible_weights_in_model
from weight_noise_ptq.compression.registry import build_compression_model


@torch.no_grad()
def evaluate_classification_loader(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """One pass: mean top-1 and mean cross-entropy."""
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    agg = ClassificationAggregate()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        agg.update(logits, y, loss)
    return {"top1": agg.mean_top1(), "loss": agg.mean_loss()}


@torch.no_grad()
def evaluate_compression_loader(
    model: nn.Module,
    loader: DataLoader[torch.Tensor],
    *,
    device: torch.device,
    lambda_rd: float,
) -> dict[str, float]:
    """Pool MSE, bpp, and RD metrics across batches with correct sample weighting.

    Batches may differ in size; MSE is pooled over all tensor elements, bpp is
    pooled with weights ``N*H*W`` per batch to match total-bits / total-pixels.
    """
    model.eval()
    sum_mse_elem: float = 0.0
    sum_el: int = 0
    sum_bpp_times_npix: float = 0.0
    sum_npix: float = 0.0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device, non_blocking=True)
        out = model(x)
        if not isinstance(out, dict) or "x_hat" not in out or "likelihoods" not in out:
            raise RuntimeError("CompressAI model forward must return dict with x_hat and likelihoods")
        x_hat = out["x_hat"]
        mse = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
        n_pix = float(x.size(0) * x.size(2) * x.size(3))
        bpp = estimate_bpp_from_likelihoods(out["likelihoods"], n_pix)
        bpp_f = float(bpp.item())
        nel = int(x.numel())
        sum_mse_elem += float(mse.item()) * float(nel)
        sum_el += nel
        sum_bpp_times_npix += bpp_f * n_pix
        sum_npix += n_pix
    if sum_el == 0 or sum_npix == 0.0:
        return {"mse": 0.0, "bpp": 0.0, "psnr": 0.0, "rd_loss": 0.0}
    mse_global = sum_mse_elem / float(sum_el)
    bpp_global = sum_bpp_times_npix / sum_npix
    psnr = psnr_from_mse(mse_global)
    # RD scalar on CPU float64 avoids device/dtype surprises when ``device`` is CUDA.
    mse_t = torch.tensor(mse_global, dtype=torch.float64, device="cpu")
    bpp_t = torch.tensor(bpp_global, dtype=torch.float64, device="cpu")
    rd_t = rate_distortion_loss(mse_t, bpp_t, lambda_rd)
    rd_global = float(rd_t.item())
    return {
        "mse": mse_global,
        "bpp": bpp_global,
        "psnr": psnr,
        "rd_loss": rd_global,
    }


def load_classification_model_from_checkpoint(
    checkpoint_path: str,
    *,
    model_name: str,
    num_classes: int,
    pretrained_arch: bool,
    device: torch.device,
) -> nn.Module:
    """Build architecture and load ``model_state_dict``."""
    m = build_classification_model(model_name, num_classes=num_classes, pretrained=pretrained_arch)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        m.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        m.load_state_dict(ckpt, strict=True)
    return m.to(device)


def load_compression_model_from_checkpoint(
    checkpoint_path: str,
    *,
    model_name: str,
    quality: int,
    metric: str,
    pretrained_arch: bool,
    device: torch.device,
) -> nn.Module:
    """Build CompressAI model and load weights."""
    m = build_compression_model(
        model_name,
        quality=quality,
        metric=metric,
        pretrained=pretrained_arch,
    )
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        m.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        m.load_state_dict(ckpt, strict=True)
    return m.to(device)


def evaluate_classification_bitwidth(
    base_model: nn.Module,
    bitwidth: str,
    *,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate classification at ``fp32`` or quantized weights (copy)."""
    if bitwidth == "fp32":
        m = base_model
    else:
        m = quantize_eligible_weights_in_model(base_model, bitwidth)
    m = m.to(device)
    return evaluate_classification_loader(m, loader, device=device)


def evaluate_compression_bitwidth(
    base_model: nn.Module,
    bitwidth: str,
    *,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    lambda_rd: float,
) -> dict[str, float]:
    """Evaluate compression at ``fp32`` or quantized weights (copy)."""
    if bitwidth == "fp32":
        m = base_model
    else:
        m = quantize_eligible_weights_in_model(base_model, bitwidth)
    m = m.to(device)
    return evaluate_compression_loader(m, loader, device=device, lambda_rd=lambda_rd)


def classification_fp32_row_payload(
    *,
    metrics: dict[str, float],
    config: dict[str, Any],
    run_dir: str,
    checkpoint: str,
    epoch_best: int,
    fp32_top1_ref: float | None = None,
) -> dict[str, Any]:
    """Build a single-row payload for ``eval_fp32.json`` (wrapped in ``rows``)."""
    top1 = float(metrics["top1"])
    loss = float(metrics["loss"])
    if fp32_top1_ref is None:
        fp32_top1_ref = top1
    drop = top1 - float(fp32_top1_ref)
    ret = top1 / float(fp32_top1_ref) if fp32_top1_ref != 0 else float("nan")
    row = {
        "task": "classification",
        "dataset": config.get("dataset", "tiny_imagenet"),
        "model": config.get("model", ""),
        "regime": config.get("regime", ""),
        "alpha": config.get("alpha", ""),
        "lambda_rd": "",
        "seed": config.get("seed", ""),
        "checkpoint": checkpoint,
        "bitwidth": "fp32",
        "quant_mode": config.get("quant_mode", ""),
        "noise_scale_mode": config.get("noise_scale_mode", ""),
        "epoch_best": epoch_best,
        "top1": top1,
        "loss": loss,
        "drop_from_fp32": drop,
        "retention_ratio": ret,
        "psnr": "",
        "bpp": "",
        "psnr_drop_from_fp32": "",
        "bpp_shift_from_fp32": "",
        "rd_loss": "",
        "run_dir": run_dir,
    }
    return {"rows": [row]}


def classification_quant_rows_payload(
    *,
    metrics_by_bitwidth: dict[str, dict[str, float]],
    config: dict[str, Any],
    run_dir: str,
    checkpoint: str,
    epoch_best: int,
    fp32_top1_ref: float,
) -> dict[str, Any]:
    """Rows for ``eval_quant.json`` (w8/w6/w4), with deltas vs FP32 top-1."""
    rows: list[dict[str, Any]] = []
    for bw in ("w8", "w6", "w4"):
        if bw not in metrics_by_bitwidth:
            continue
        m = metrics_by_bitwidth[bw]
        top1 = float(m["top1"])
        loss = float(m["loss"])
        drop = top1 - fp32_top1_ref
        ret = top1 / fp32_top1_ref if fp32_top1_ref != 0 else float("nan")
        rows.append(
            {
                "task": "classification",
                "dataset": config.get("dataset", "tiny_imagenet"),
                "model": config.get("model", ""),
                "regime": config.get("regime", ""),
                "alpha": config.get("alpha", ""),
                "lambda_rd": "",
                "seed": config.get("seed", ""),
                "checkpoint": checkpoint,
                "bitwidth": bw,
                "quant_mode": config.get("quant_mode", ""),
                "noise_scale_mode": config.get("noise_scale_mode", ""),
                "epoch_best": epoch_best,
                "top1": top1,
                "loss": loss,
                "drop_from_fp32": drop,
                "retention_ratio": ret,
                "psnr": "",
                "bpp": "",
                "psnr_drop_from_fp32": "",
                "bpp_shift_from_fp32": "",
                "rd_loss": "",
                "run_dir": run_dir,
            },
        )
    return {"rows": rows}


def compression_quant_rows_payload(
    *,
    metrics_by_bitwidth: dict[str, dict[str, float]],
    config: dict[str, Any],
    run_dir: str,
    checkpoint: str,
    epoch_best: int,
    fp32_psnr_ref: float,
    fp32_bpp_ref: float,
) -> dict[str, Any]:
    """Rows for ``eval_quant.json`` (w8/w6/w4), with deltas vs FP32 PSNR/BPP."""
    rows: list[dict[str, Any]] = []
    for bw in ("w8", "w6", "w4"):
        if bw not in metrics_by_bitwidth:
            continue
        m = metrics_by_bitwidth[bw]
        psnr = float(m["psnr"])
        bpp = float(m["bpp"])
        rd = float(m["rd_loss"])
        rows.append(
            {
                "task": "compression",
                "dataset": config.get("dataset", "tiny_imagenet"),
                "model": config.get("model", ""),
                "regime": config.get("regime", ""),
                "alpha": config.get("alpha", ""),
                "lambda_rd": config.get("lambda_rd", ""),
                "seed": config.get("seed", ""),
                "checkpoint": checkpoint,
                "bitwidth": bw,
                "quant_mode": config.get("quant_mode", ""),
                "noise_scale_mode": config.get("noise_scale_mode", ""),
                "epoch_best": epoch_best,
                "top1": "",
                "loss": "",
                "drop_from_fp32": "",
                "retention_ratio": "",
                "psnr": psnr,
                "bpp": bpp,
                "psnr_drop_from_fp32": psnr - fp32_psnr_ref,
                "bpp_shift_from_fp32": bpp - fp32_bpp_ref,
                "rd_loss": rd,
                "run_dir": run_dir,
            },
        )
    return {"rows": rows}


def compression_fp32_row_payload(
    *,
    metrics: dict[str, float],
    config: dict[str, Any],
    run_dir: str,
    checkpoint: str,
    epoch_best: int,
    fp32_psnr_ref: float | None = None,
    fp32_bpp_ref: float | None = None,
) -> dict[str, Any]:
    """Build a single-row payload for compression ``eval_fp32.json``."""
    psnr = float(metrics["psnr"])
    bpp = float(metrics["bpp"])
    rd = float(metrics["rd_loss"])
    if fp32_psnr_ref is None:
        fp32_psnr_ref = psnr
    if fp32_bpp_ref is None:
        fp32_bpp_ref = bpp
    row = {
        "task": "compression",
        "dataset": config.get("dataset", "tiny_imagenet"),
        "model": config.get("model", ""),
        "regime": config.get("regime", ""),
        "alpha": config.get("alpha", ""),
        "lambda_rd": config.get("lambda_rd", ""),
        "seed": config.get("seed", ""),
        "checkpoint": checkpoint,
        "bitwidth": "fp32",
        "quant_mode": config.get("quant_mode", ""),
        "noise_scale_mode": config.get("noise_scale_mode", ""),
        "epoch_best": epoch_best,
        "top1": "",
        "loss": "",
        "drop_from_fp32": "",
        "retention_ratio": "",
        "psnr": psnr,
        "bpp": bpp,
        "psnr_drop_from_fp32": psnr - float(fp32_psnr_ref),
        "bpp_shift_from_fp32": bpp - float(fp32_bpp_ref),
        "rd_loss": rd,
        "run_dir": run_dir,
    }
    return {"rows": [row]}
