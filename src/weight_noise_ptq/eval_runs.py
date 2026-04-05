"""High-level FP32 and quantized evaluation entrypoints (used by CLI scripts)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from weight_noise_ptq.classification.datasets import (
    TinyImageNetClassificationDataset,
    val_transforms_224,
)
from weight_noise_ptq.common.checkpointing import load_checkpoint
from weight_noise_ptq.common.config import ClassificationConfig, CompressionConfig
from weight_noise_ptq.common.logging_utils import save_json
from weight_noise_ptq.common.paths import classification_run_dir, compression_run_dir
from weight_noise_ptq.compression.datasets import TinyImageNetCompressionDataset, val_transforms_64
from weight_noise_ptq.common.device_utils import resolve_torch_device
from weight_noise_ptq.eval_helpers import (
    classification_fp32_row_payload,
    classification_quant_rows_payload,
    compression_fp32_row_payload,
    compression_quant_rows_payload,
    evaluate_classification_bitwidth,
    evaluate_classification_loader,
    evaluate_compression_bitwidth,
    evaluate_compression_loader,
    load_classification_model_from_checkpoint,
    load_compression_model_from_checkpoint,
)


def _iter_eval_json_rows(data: Any) -> list[dict[str, Any]]:
    """Flatten eval JSON payloads to a list of row dicts (same shapes as :mod:`results_export`)."""
    if isinstance(data, dict) and "rows" in data and isinstance(data["rows"], list):
        return [sub for sub in data["rows"] if isinstance(sub, dict)]
    if isinstance(data, list):
        return [sub for sub in data if isinstance(sub, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _find_fp32_baseline_row(data: Any) -> dict[str, Any]:
    """Return the single ``bitwidth == fp32`` row; quantized deltas require this baseline."""
    rows = _iter_eval_json_rows(data)
    for row in rows:
        if str(row.get("bitwidth")) == "fp32":
            return row
    raise ValueError(
        f"No row with bitwidth 'fp32' in eval_fp32 payload ({len(rows)} row(s) present). "
        "Run FP32 eval for this run before quant eval."
    )


def _assert_fp32_row_matches_quant_config(
    row: dict[str, Any],
    *,
    expected_task: str,
    model: str,
    regime: str,
    seed: int,
    eval_fp32_path: Path,
) -> None:
    """Ensure the on-disk FP32 baseline matches the YAML/CLI quant run (no silent cross-run mixing)."""
    if str(row.get("task", "")) != expected_task:
        raise ValueError(
            f"{eval_fp32_path}: fp32 row has task {row.get('task')!r}, expected {expected_task!r}. "
            "Refusing quant eval so metrics are not compared to the wrong baseline."
        )
    if str(row.get("model", "")) != str(model):
        raise ValueError(
            f"{eval_fp32_path}: fp32 row model {row.get('model')!r} != config model {model!r}."
        )
    if str(row.get("regime", "")) != str(regime):
        raise ValueError(
            f"{eval_fp32_path}: fp32 row regime {row.get('regime')!r} != config regime {regime!r}."
        )
    if int(row.get("seed", -10**9)) != int(seed):
        raise ValueError(
            f"{eval_fp32_path}: fp32 row seed {row.get('seed')!r} != requested seed {seed}."
        )


def run_eval_fp32_classification(
    cfg: ClassificationConfig,
    *,
    seed: int,
    checkpoint_name: str,
    data_root: Path | str | None,
    device: str | torch.device | None,
    num_workers: int | None,
    results_base: Path | None,
) -> Path:
    """Evaluate ``best`` or ``last`` checkpoint at FP32; write ``eval_fp32.json``."""
    dev = resolve_torch_device(device)
    droot = Path(data_root) if data_root is not None else Path(cfg.data_root)
    nw = int(num_workers) if num_workers is not None else int(cfg.num_workers)
    run_dir = classification_run_dir(cfg.model, cfg.regime, seed, results_base=results_base)
    ckpt_path = run_dir / f"{checkpoint_name}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model = load_classification_model_from_checkpoint(
        str(ckpt_path),
        model_name=cfg.model,
        num_classes=cfg.num_classes,
        pretrained_arch=cfg.pretrained_backbone,
        device=dev,
    )
    ds = TinyImageNetClassificationDataset(droot, split="val", transform=val_transforms_224())
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=dev.type == "cuda",
    )
    metrics = evaluate_classification_loader(model, loader, device=dev)
    _, meta = load_checkpoint(ckpt_path, map_location=dev)
    epoch_best = int(meta.epoch) if meta is not None else -1

    cfg_dict = asdict(cfg)
    cfg_dict["seed"] = int(seed)
    payload = classification_fp32_row_payload(
        metrics=metrics,
        config=cfg_dict,
        run_dir=str(run_dir.resolve()),
        checkpoint=checkpoint_name,
        epoch_best=epoch_best,
    )
    save_json(run_dir / "eval_fp32.json", payload)
    return run_dir


def run_eval_fp32_compression(
    cfg: CompressionConfig,
    *,
    seed: int,
    checkpoint_name: str,
    data_root: Path | str | None,
    device: str | torch.device | None,
    num_workers: int | None,
    results_base: Path | None,
) -> Path:
    """Evaluate FP32 compression metrics; write ``eval_fp32.json``."""
    dev = resolve_torch_device(device)
    droot = Path(data_root) if data_root is not None else Path(cfg.data_root)
    nw = int(num_workers) if num_workers is not None else int(cfg.num_workers)
    run_dir = compression_run_dir(cfg.model, cfg.regime, seed, results_base=results_base)
    ckpt_path = run_dir / f"{checkpoint_name}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model = load_compression_model_from_checkpoint(
        str(ckpt_path),
        model_name=cfg.model,
        quality=cfg.compressai_quality,
        metric=cfg.compressai_metric,
        pretrained_arch=cfg.compressai_pretrained,
        device=dev,
    )
    ds = TinyImageNetCompressionDataset(droot, split="val", transform=val_transforms_64())
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=dev.type == "cuda",
    )
    metrics = evaluate_compression_loader(model, loader, device=dev, lambda_rd=cfg.lambda_rd)
    _, meta = load_checkpoint(ckpt_path, map_location=dev)
    epoch_best = int(meta.epoch) if meta is not None else -1

    cfg_dict = asdict(cfg)
    cfg_dict["seed"] = int(seed)
    payload = compression_fp32_row_payload(
        metrics=metrics,
        config=cfg_dict,
        run_dir=str(run_dir.resolve()),
        checkpoint=checkpoint_name,
        epoch_best=epoch_best,
    )
    save_json(run_dir / "eval_fp32.json", payload)
    return run_dir


def run_eval_quant_classification(
    cfg: ClassificationConfig,
    *,
    seed: int,
    checkpoint_name: str,
    data_root: Path | str | None,
    device: str | torch.device | None,
    num_workers: int | None,
    results_base: Path | None,
) -> Path:
    """Quantized eval (w8/w6/w4); requires existing ``eval_fp32.json`` for FP32 reference top-1."""
    dev = resolve_torch_device(device)
    droot = Path(data_root) if data_root is not None else Path(cfg.data_root)
    nw = int(num_workers) if num_workers is not None else int(cfg.num_workers)
    run_dir = classification_run_dir(cfg.model, cfg.regime, seed, results_base=results_base)
    ckpt_path = run_dir / f"{checkpoint_name}.pt"
    eval_fp32_path = run_dir / "eval_fp32.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not eval_fp32_path.is_file():
        raise FileNotFoundError(f"Missing {eval_fp32_path}; run FP32 eval first.")

    from weight_noise_ptq.common.logging_utils import load_json

    fp32_data = load_json(eval_fp32_path)
    fp32_row = _find_fp32_baseline_row(fp32_data)
    _assert_fp32_row_matches_quant_config(
        fp32_row,
        expected_task="classification",
        model=cfg.model,
        regime=cfg.regime,
        seed=seed,
        eval_fp32_path=eval_fp32_path,
    )
    fp32_top1 = float(fp32_row["top1"])

    model = load_classification_model_from_checkpoint(
        str(ckpt_path),
        model_name=cfg.model,
        num_classes=cfg.num_classes,
        pretrained_arch=cfg.pretrained_backbone,
        device=dev,
    )
    ds = TinyImageNetClassificationDataset(droot, split="val", transform=val_transforms_224())
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=dev.type == "cuda",
    )
    metrics_by_bw: dict[str, dict[str, float]] = {}
    for bw in ("w8", "w6", "w4"):
        metrics_by_bw[bw] = evaluate_classification_bitwidth(
            model,
            bw,
            loader=loader,
            device=dev,
        )
    _, meta = load_checkpoint(ckpt_path, map_location=dev)
    epoch_best = int(meta.epoch) if meta is not None else -1

    cfg_dict = asdict(cfg)
    cfg_dict["seed"] = int(seed)
    payload = classification_quant_rows_payload(
        metrics_by_bitwidth=metrics_by_bw,
        config=cfg_dict,
        run_dir=str(run_dir.resolve()),
        checkpoint=checkpoint_name,
        epoch_best=epoch_best,
        fp32_top1_ref=fp32_top1,
    )
    save_json(run_dir / "eval_quant.json", payload)
    return run_dir


def run_eval_quant_compression(
    cfg: CompressionConfig,
    *,
    seed: int,
    checkpoint_name: str,
    data_root: Path | str | None,
    device: str | torch.device | None,
    num_workers: int | None,
    results_base: Path | None,
) -> Path:
    """Quantized compression eval; requires ``eval_fp32.json`` for FP32 PSNR/BPP refs."""
    dev = resolve_torch_device(device)
    droot = Path(data_root) if data_root is not None else Path(cfg.data_root)
    nw = int(num_workers) if num_workers is not None else int(cfg.num_workers)
    run_dir = compression_run_dir(cfg.model, cfg.regime, seed, results_base=results_base)
    ckpt_path = run_dir / f"{checkpoint_name}.pt"
    eval_fp32_path = run_dir / "eval_fp32.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not eval_fp32_path.is_file():
        raise FileNotFoundError(f"Missing {eval_fp32_path}; run FP32 eval first.")

    from weight_noise_ptq.common.logging_utils import load_json

    fp32_data = load_json(eval_fp32_path)
    fp32_row = _find_fp32_baseline_row(fp32_data)
    _assert_fp32_row_matches_quant_config(
        fp32_row,
        expected_task="compression",
        model=cfg.model,
        regime=cfg.regime,
        seed=seed,
        eval_fp32_path=eval_fp32_path,
    )
    fp32_psnr = float(fp32_row["psnr"])
    fp32_bpp = float(fp32_row["bpp"])

    model = load_compression_model_from_checkpoint(
        str(ckpt_path),
        model_name=cfg.model,
        quality=cfg.compressai_quality,
        metric=cfg.compressai_metric,
        pretrained_arch=cfg.compressai_pretrained,
        device=dev,
    )
    ds = TinyImageNetCompressionDataset(droot, split="val", transform=val_transforms_64())
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=dev.type == "cuda",
    )
    metrics_by_bw: dict[str, dict[str, float]] = {}
    for bw in ("w8", "w6", "w4"):
        metrics_by_bw[bw] = evaluate_compression_bitwidth(
            model,
            bw,
            loader=loader,
            device=dev,
            lambda_rd=cfg.lambda_rd,
        )
    _, meta = load_checkpoint(ckpt_path, map_location=dev)
    epoch_best = int(meta.epoch) if meta is not None else -1

    cfg_dict = asdict(cfg)
    cfg_dict["seed"] = int(seed)
    payload = compression_quant_rows_payload(
        metrics_by_bitwidth=metrics_by_bw,
        config=cfg_dict,
        run_dir=str(run_dir.resolve()),
        checkpoint=checkpoint_name,
        epoch_best=epoch_best,
        fp32_psnr_ref=fp32_psnr,
        fp32_bpp_ref=fp32_bpp,
    )
    save_json(run_dir / "eval_quant.json", payload)
    return run_dir
