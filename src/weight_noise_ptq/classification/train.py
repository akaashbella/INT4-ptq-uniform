"""Tiny ImageNet classification training (clean vs uniform weight noise).

Artifacts written here: ``config.json``, ``environment.json``, ``train_log.csv``,
``val_log.csv``, ``best.pt``, ``last.pt``. Run ``scripts/eval_fp32.py`` and
``scripts/eval_quant.py`` after training to produce ``eval_fp32.json`` and
``eval_quant.json``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from weight_noise_ptq.classification.datasets import (
    TinyImageNetClassificationDataset,
    train_transforms_224,
    val_transforms_224,
)
from weight_noise_ptq.classification.registry import build_classification_model
from weight_noise_ptq.common.checkpointing import (
    CheckpointMetadata,
    save_checkpoint,
    verify_training_run_artifacts,
)
from weight_noise_ptq.common.device_utils import resolve_torch_device
from weight_noise_ptq.common.config import ClassificationConfig
from weight_noise_ptq.common.environment import collect_environment_metadata
from weight_noise_ptq.common.logging_utils import TrainLogWriter, ValLogWriter, save_json
from weight_noise_ptq.common.metrics import ClassificationAggregate
from weight_noise_ptq.common.noise import temporary_uniform_weight_noise
from weight_noise_ptq.common.optimizers import build_optimizer
from weight_noise_ptq.common.paths import classification_run_dir, repo_root
from weight_noise_ptq.common.seed import set_seed, worker_init_fn
from weight_noise_ptq.common.validators import validate_classification_config

logger = logging.getLogger(__name__)




@torch.no_grad()
def _validate_classification(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return (mean top1, mean cross-entropy)."""
    model.eval()
    agg = ClassificationAggregate()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        agg.update(logits, y, loss)
    return agg.mean_top1(), agg.mean_loss()


def train_classification(
    cfg: ClassificationConfig,
    *,
    seed: int,
    data_root: Path | str | None = None,
    device: torch.device | str | None = None,
    num_workers: int | None = None,
    results_base: Path | None = None,
    repo_root_override: Path | str | None = None,
) -> Path:
    """Run full training; return the run directory."""
    validate_classification_config(cfg)
    set_seed(int(seed))

    droot = Path(data_root) if data_root is not None else Path(cfg.data_root)
    dev = resolve_torch_device(device if isinstance(device, str) else (device if device is not None else None))
    nw = int(num_workers) if num_workers is not None else int(cfg.num_workers)

    run_dir = classification_run_dir(cfg.model, cfg.regime, seed, results_base=results_base)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Fresh CSV logs
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

    train_ds = TinyImageNetClassificationDataset(
        droot,
        split="train",
        transform=train_transforms_224(),
    )
    val_ds = TinyImageNetClassificationDataset(
        droot,
        split="val",
        transform=val_transforms_224(),
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

    model = build_classification_model(
        cfg.model,
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained_backbone,
    ).to(dev)
    optimizer = build_optimizer(model, cfg.optim)
    criterion = nn.CrossEntropyLoss()

    train_log = TrainLogWriter(run_dir / "train_log.csv")
    val_log = ValLogWriter(run_dir / "val_log.csv")

    regime = cfg.regime
    use_noise = regime == "noisy_uniform_a0.02"
    alpha = float(cfg.alpha) if use_noise else 0.0

    best_top1 = -1.0
    best_epoch = -1

    logger.info(
        "Starting classification: model=%s regime=%s seed=%s epochs=%s device=%s data_root=%s",
        cfg.model,
        regime,
        seed,
        cfg.epochs,
        dev,
        droot,
    )

    global_step = 0
    for epoch in range(int(cfg.epochs)):
        model.train()
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(
            train_loader,
            desc=f"train e{epoch + 1}/{cfg.epochs}",
            leave=False,
        )
        for x, y in pbar:
            global_step += 1
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            lr = float(optimizer.param_groups[0]["lr"])

            if use_noise:
                with temporary_uniform_weight_noise(model, alpha=alpha):
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()

            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1
            train_log.writerow(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "lr": lr,
                    "loss": float(loss.item()),
                    "regime": regime,
                },
            )
            pbar.set_postfix(loss=float(loss.item()))

        mean_train_loss = running_loss / max(n_batches, 1)

        val_top1, val_loss = _validate_classification(model, val_loader, criterion, dev)
        val_log.writerow(
            {
                "epoch": epoch + 1,
                "metric_name": "top1",
                "metric_value": val_top1,
                "regime": "clean",
            },
        )
        val_log.writerow(
            {
                "epoch": epoch + 1,
                "metric_name": "cross_entropy",
                "metric_value": val_loss,
                "regime": "clean",
            },
        )

        logger.info(
            "Epoch %s/%s train_loss=%.4f val_top1=%.4f val_loss=%.4f",
            epoch + 1,
            cfg.epochs,
            mean_train_loss,
            val_top1,
            val_loss,
        )

        last_meta = CheckpointMetadata(
            task="classification",
            model=cfg.model,
            regime=cfg.regime,
            seed=int(seed),
            epoch=epoch + 1,
            kind="last",
            metric_name="val_top1",
            metric_value=float(val_top1),
            extra={"val_loss": float(val_loss), "mean_train_loss": mean_train_loss},
        )
        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            metadata=last_meta,
        )

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_epoch = epoch + 1
            best_meta = CheckpointMetadata(
                task="classification",
                model=cfg.model,
                regime=cfg.regime,
                seed=int(seed),
                epoch=epoch + 1,
                kind="best",
                metric_name="val_top1",
                metric_value=float(val_top1),
                extra={"val_loss": float(val_loss)},
            )
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                metadata=best_meta,
            )

    logger.info("Finished. best val_top1=%.4f at epoch %s (run_dir=%s)", best_top1, best_epoch, run_dir)
    verify_training_run_artifacts(run_dir)
    return run_dir
