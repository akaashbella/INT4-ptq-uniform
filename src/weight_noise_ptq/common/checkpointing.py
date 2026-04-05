"""Checkpoint save/load for ``best.pt`` and ``last.pt`` with task-aware metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

CheckpointKind = Literal["best", "last"]
TaskKind = Literal["classification", "compression"]


@dataclass
class CheckpointMetadata:
    """Structured metadata stored inside checkpoint files."""

    task: TaskKind
    model: str
    regime: str
    seed: int
    epoch: int
    kind: CheckpointKind
    metric_name: str
    metric_value: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    metadata: CheckpointMetadata,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Atomically save model weights, optional optimizer, and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata.to_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra_state:
        payload["extra"] = extra_state
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[dict[str, Any], CheckpointMetadata | None]:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Returns ``(raw_dict, metadata)``. Older checkpoints without metadata return
    ``metadata=None``.
    """
    path = Path(path)
    try:
        raw = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location=map_location)
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected checkpoint format in {path}")
    meta_raw = raw.get("metadata")
    meta: CheckpointMetadata | None = None
    if isinstance(meta_raw, dict):
        try:
            meta = CheckpointMetadata(
                task=meta_raw["task"],
                model=meta_raw["model"],
                regime=meta_raw["regime"],
                seed=int(meta_raw["seed"]),
                epoch=int(meta_raw["epoch"]),
                kind=meta_raw["kind"],
                metric_name=str(meta_raw["metric_name"]),
                metric_value=float(meta_raw["metric_value"]),
                extra=dict(meta_raw.get("extra", {})),
            )
        except (KeyError, TypeError, ValueError):
            meta = None
    return raw, meta


def load_model_state_dict(path: Path, *, map_location: str | torch.device | None = None) -> dict[str, Any]:
    """Return only ``model_state_dict`` from a checkpoint."""
    raw, _ = load_checkpoint(path, map_location=map_location)
    if "model_state_dict" not in raw:
        raise KeyError(f"No model_state_dict in {path}")
    return raw["model_state_dict"]


# Files a successful train_* run must leave behind (before eval scripts add eval_*.json).
TRAINING_RUN_REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "environment.json",
    "train_log.csv",
    "val_log.csv",
    "best.pt",
    "last.pt",
)


def verify_training_run_artifacts(run_dir: Path) -> None:
    """Raise if the run directory is missing any required training artifact."""
    run_dir = Path(run_dir)
    missing = [name for name in TRAINING_RUN_REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Training run directory {run_dir} is missing required files: {missing}. "
            "Downstream FP32/quant eval expects a complete run."
        )
