"""Tiny ImageNet 64×64 compression training (MSE rate–distortion + optional weight noise).

Writes checkpoints and CSV logs under the run directory; run ``scripts/eval_fp32.py`` and
``scripts/eval_quant.py`` afterward for ``eval_*.json``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from weight_noise_ptq.common.checkpointing import (
    CheckpointMetadata,
    save_checkpoint,
    verify_training_run_artifacts,
)
from weight_noise_ptq.common.device_utils import resolve_torch_device
from weight_noise_ptq.common.config import CompressionConfig
from weight_noise_ptq.common.environment import collect_environment_metadata
from weight_noise_ptq.common.logging_utils import TrainLogWriter, ValLogWriter, save_json
from weight_noise_ptq.common.metrics import estimate_bpp_from_likelihoods, rate_distortion_loss
from weight_noise_ptq.common.noise import temporary_uniform_weight_noise
from weight_noise_ptq.common.optimizers import build_optimizer
from weight_noise_ptq.common.paths import compression_run_dir, repo_root
from weight_noise_ptq.common.seed import set_seed, worker_init_fn
from weight_noise_ptq.common.validators import validate_compression_config
from weight_noise_ptq.compression.datasets import (
    TinyImageNetCompressionDataset,
    train_transforms_64,
    val_transforms_64,
)
from weight_noise_ptq.compression.registry import build_compression_model
from weight_noise_ptq.eval_helpers import evaluate_compression_loader

logger = logging.getLogger(__name__)


def _compression_train_loss(model: nn.Module, x: torch.Tensor, lambda_rd: float) -> torch.Tensor:
    """Standard MSE RD plus CompressAI entropy-bottleneck auxiliary loss (training only)."""
    out = model(x)
    if not isinstance(out, dict) or "x_hat" not in out or "likelihoods" not in out:
        raise RuntimeError("CompressAI forward must return x_hat and likelihoods")
    mse = torch.nn.functional.mse_loss(out["x_hat"], x)
    n_pix = x.size(0) * x.size(2) * x.size(3)
    bpp = estimate_bpp_from_likelihoods(out["likelihoods"], n_pix)
    rd = rate_distortion_loss(mse, bpp, lambda_rd)
    total: torch.Tensor = rd
    if hasattr(model, "aux_loss"):
        total = total + model.aux_loss()
    return total


def train_compression(
    cfg: CompressionConfig,
    *,
    seed: int,
    data_root: Path | str | None = None,
    device: torch.device | str | None = None,
    num_workers: int | None = None,
    results_base: Path | None = None,
    repo_root_override: Path | str | None = None,
) -> Path:
    """Run full training; return the run directory."""
    validate_compression_config(cfg)
    set_seed(int(seed))

    droot = Path(data_root) if data_root is not None else Path(cfg.data_root)
    dev = resolve_torch_device(device if isinstance(device, str) else (device if device is not None else None))
    nw = int(num_workers) if num_workers is not None else int(cfg.num_workers)
    lambda_rd = float(cfg.lambda_rd)

    run_dir = compression_run_dir(cfg.model, cfg.regime, seed, results_base=results_base)
    run_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train_log.csv", "val_log.csv"):
        p = run_dir / name
        if p.exists():
            p.unlink()

    repo = Path(repo_root_override) if repo_root_override is not None else repo_root()

    cfg_payload = asdict(cfg)
    cfg_payload["seed"] = int(seed)
    save_json(run_dir / "config.json", cfg_payload)
    save_json(
        run_dir / "environment.json",
        collect_environment_metadata(repo_root_path=str(repo)),
    )

    train_ds = TinyImageNetCompressionDataset(
        droot,
        split="train",
        transform=train_transforms_64(),
    )
    val_ds = TinyImageNetCompressionDataset(
        droot,
        split="val",
        transform=val_transforms_64(),
    )

    def _worker_init(wid: int) -> None:
        worker_init_fn(wid, int(seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=dev.type == "cuda",
        worker_init_fn=_worker_init if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=dev.type == "cuda",
    )

    model = build_compression_model(
        cfg.model,
        quality=cfg.compressai_quality,
        metric=cfg.compressai_metric,
        pretrained=cfg.compressai_pretrained,
    ).to(dev)
    optimizer = build_optimizer(model, cfg.optim)

    train_log = TrainLogWriter(run_dir / "train_log.csv")
    val_log = ValLogWriter(run_dir / "val_log.csv")

    regime = cfg.regime
    use_noise = regime == "noisy_uniform_a0.02"
    alpha = float(cfg.alpha) if use_noise else 0.0

    best_rd = float("inf")
    best_epoch = -1

    logger.info(
        "Starting compression: model=%s regime=%s seed=%s epochs=%s lambda_rd=%s device=%s data_root=%s",
        cfg.model,
        regime,
        seed,
        cfg.epochs,
        lambda_rd,
        dev,
        droot,
    )

    global_step = 0
    for epoch in range(int(cfg.epochs)):
        model.train()
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"train e{epoch + 1}/{cfg.epochs}", leave=False)
        for x in pbar:
            global_step += 1
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            lr = float(optimizer.param_groups[0]["lr"])

            if use_noise:
                with temporary_uniform_weight_noise(model, alpha=alpha):
                    loss = _compression_train_loss(model, x, lambda_rd)
                    loss.backward()
            else:
                loss = _compression_train_loss(model, x, lambda_rd)
                loss.backward()

            optimizer.step()

            li = float(loss.item())
            running_loss += li
            n_batches += 1
            train_log.writerow(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "lr": lr,
                    "loss": li,
                    "regime": regime,
                },
            )
            pbar.set_postfix(loss=li)

        mean_train_loss = running_loss / max(n_batches, 1)

        model.eval()
        val_metrics = evaluate_compression_loader(model, val_loader, device=dev, lambda_rd=lambda_rd)
        val_rd = float(val_metrics["rd_loss"])
        val_psnr = float(val_metrics["psnr"])
        val_bpp = float(val_metrics["bpp"])

        val_log.writerow(
            {
                "epoch": epoch + 1,
                "metric_name": "rd_loss",
                "metric_value": val_rd,
                "regime": "clean",
            },
        )
        val_log.writerow(
            {
                "epoch": epoch + 1,
                "metric_name": "psnr",
                "metric_value": val_psnr,
                "regime": "clean",
            },
        )
        val_log.writerow(
            {
                "epoch": epoch + 1,
                "metric_name": "bpp",
                "metric_value": val_bpp,
                "regime": "clean",
            },
        )

        logger.info(
            "Epoch %s/%s train_loss=%.6f val_rd_loss=%.6f val_psnr=%.4f val_bpp=%.6f",
            epoch + 1,
            cfg.epochs,
            mean_train_loss,
            val_rd,
            val_psnr,
            val_bpp,
        )

        last_meta = CheckpointMetadata(
            task="compression",
            model=cfg.model,
            regime=cfg.regime,
            seed=int(seed),
            epoch=epoch + 1,
            kind="last",
            metric_name="val_rd_loss",
            metric_value=val_rd,
            extra={"psnr": val_psnr, "bpp": val_bpp, "mean_train_loss": mean_train_loss},
        )
        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            metadata=last_meta,
        )

        if val_rd < best_rd:
            best_rd = val_rd
            best_epoch = epoch + 1
            best_meta = CheckpointMetadata(
                task="compression",
                model=cfg.model,
                regime=cfg.regime,
                seed=int(seed),
                epoch=epoch + 1,
                kind="best",
                metric_name="val_rd_loss",
                metric_value=val_rd,
                extra={"psnr": val_psnr, "bpp": val_bpp},
            )
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                metadata=best_meta,
            )

    logger.info("Finished. best val_rd_loss=%.6f at epoch %s (run_dir=%s)", best_rd, best_epoch, run_dir)
    verify_training_run_artifacts(run_dir)
    return run_dir
