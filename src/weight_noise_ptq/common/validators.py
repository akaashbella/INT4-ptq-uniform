"""Validate locked experiment names and consistency (fail fast on bad configs)."""

from __future__ import annotations

from typing import FrozenSet

from weight_noise_ptq.common.config import ClassificationConfig, CompressionConfig
from weight_noise_ptq.common.locked_names import (
    BITWIDTH_LABELS,
    CLASSIFICATION_MODELS as _CLS_TUPLE,
    COMPRESSION_MODELS as _CMP_TUPLE,
    LOCKED_TRAIN_EPOCHS,
    REGIMES as _REG_TUPLE,
    RUN_SEEDS,
)

CLASSIFICATION_MODELS: FrozenSet[str] = frozenset(_CLS_TUPLE)
COMPRESSION_MODELS: FrozenSet[str] = frozenset(_CMP_TUPLE)
REGIMES: FrozenSet[str] = frozenset(_REG_TUPLE)
BITWIDTHS: FrozenSet[str] = frozenset(BITWIDTH_LABELS)


def validate_regime_and_alpha(regime: str, alpha: float) -> None:
    """``clean`` must use ``alpha == 0``; ``noisy_uniform_a0.02`` must use ``alpha == 0.02``."""
    if regime not in REGIMES:
        raise ValueError(f"Invalid regime {regime!r}; expected one of {sorted(REGIMES)}")
    if regime == "clean" and float(alpha) != 0.0:
        raise ValueError("regime 'clean' requires alpha == 0.0")
    if regime == "noisy_uniform_a0.02" and abs(float(alpha) - 0.02) > 1e-9:
        raise ValueError("regime 'noisy_uniform_a0.02' requires alpha == 0.02")


def validate_bitwidths_eval(labels: list[str]) -> None:
    for b in labels:
        if b not in BITWIDTHS:
            raise ValueError(f"Invalid bitwidth {b!r}; expected one of {sorted(BITWIDTHS)}")


def _validate_locked_epochs_and_seeds(cfg: ClassificationConfig | CompressionConfig) -> None:
    if int(cfg.epochs) != LOCKED_TRAIN_EPOCHS:
        raise ValueError(
            f"Locked training epochs are {LOCKED_TRAIN_EPOCHS}; got {cfg.epochs!r}. "
            "Change configs only for explicit off-matrix debugging.",
        )
    if tuple(cfg.seeds) != RUN_SEEDS:
        raise ValueError(
            f"Locked run seeds are {list(RUN_SEEDS)}; got {list(cfg.seeds)!r}.",
        )


def validate_classification_config(cfg: ClassificationConfig) -> None:
    if cfg.task != "classification":
        raise ValueError(f"Expected task 'classification', got {cfg.task!r}")
    if cfg.dataset != "tiny_imagenet":
        raise ValueError(f"Locked dataset is tiny_imagenet; got {cfg.dataset!r}")
    if cfg.model not in CLASSIFICATION_MODELS:
        raise ValueError(f"Invalid classification model {cfg.model!r}; expected {sorted(CLASSIFICATION_MODELS)}")
    if int(cfg.num_classes) != 200:
        raise ValueError(f"Tiny ImageNet requires num_classes=200; got {cfg.num_classes}")
    validate_regime_and_alpha(cfg.regime, cfg.alpha)
    validate_bitwidths_eval(list(cfg.bitwidths_eval))
    _validate_locked_epochs_and_seeds(cfg)


def validate_compression_config(cfg: CompressionConfig) -> None:
    if cfg.task != "compression":
        raise ValueError(f"Expected task 'compression', got {cfg.task!r}")
    if cfg.dataset != "tiny_imagenet":
        raise ValueError(f"Locked dataset is tiny_imagenet; got {cfg.dataset!r}")
    if cfg.model not in COMPRESSION_MODELS:
        raise ValueError(f"Invalid compression model {cfg.model!r}; expected {sorted(COMPRESSION_MODELS)}")
    validate_regime_and_alpha(cfg.regime, cfg.alpha)
    validate_bitwidths_eval(list(cfg.bitwidths_eval))
    _validate_locked_epochs_and_seeds(cfg)
    if abs(float(cfg.lambda_rd) - 0.0130) > 1e-5:
        raise ValueError(f"Locked lambda_rd is 0.0130; got {cfg.lambda_rd}")
