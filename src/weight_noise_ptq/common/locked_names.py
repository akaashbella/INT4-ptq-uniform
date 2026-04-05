"""Single source of truth for locked experiment identifiers (configs, paths, launchers).

All registries, validators, and ``scripts/launch_commands.py`` must stay aligned with
these tuples — change here first, then configs and docs.
"""

from __future__ import annotations

# Classification (TorchVision)
CLASSIFICATION_MODELS: tuple[str, ...] = ("resnet50", "mobilenetv3_large", "convnext_tiny")

# Compression (CompressAI registry wrappers)
COMPRESSION_MODELS: tuple[str, ...] = ("factorized_prior", "scale_hyperprior", "cheng2020_attention")

REGIMES: tuple[str, ...] = ("clean", "noisy_uniform_a0.02")

RUN_SEEDS: tuple[int, ...] = (0, 1, 2)

# Post-training eval bitwidth labels (master CSV + eval JSON)
BITWIDTH_LABELS: tuple[str, ...] = ("fp32", "w8", "w6", "w4")

# YAML filename pattern: configs/<task_dir>/{model}_{regime}.yaml
def classification_config_stem(model: str, regime: str) -> str:
    return f"{model}_{regime}"


def compression_config_stem(model: str, regime: str) -> str:
    return f"{model}_{regime}"
