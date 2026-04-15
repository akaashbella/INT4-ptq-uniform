"""Structured experiment configuration (YAML + dataclasses)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Type, TypeVar

import yaml

T = TypeVar("T")

TaskName = Literal["classification", "compression"]
DatasetName = Literal["tiny_imagenet"]
RegimeName = Literal["clean", "noisy_uniform_a0.02"]

ClassificationModel = Literal["resnet50", "mobilenetv3_large", "convnext_tiny"]
CompressionModel = Literal["factorized_prior", "scale_hyperprior", "cheng2020_attention"]

Bitwidth = Literal["fp32", "w8", "w6", "w4"]


@dataclass
class OptimConfig:
    """Optimizer hyperparameters (task trainers may interpret these)."""

    name: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4


@dataclass
class SharedExperimentConfig:
    """Fields common to classification and compression."""

    task: TaskName
    dataset: DatasetName
    model: str
    regime: RegimeName
    # Locked noisy default; ``clean`` configs must set ``alpha: 0.0`` explicitly.
    alpha: float = 0.02
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    results_dir: str = "results"
    data_root: str = "data/tiny-imagenet-200"
    bitwidths_eval: list[str] = field(
        default_factory=lambda: ["fp32", "w8", "w6", "w4"],
    )
    quant_mode: str = "symmetric_per_tensor"
    noise_scale_mode: str = "per_tensor_std"
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class ClassificationConfig(SharedExperimentConfig):
    """Tiny ImageNet classification at 224×224."""

    num_classes: int = 200
    pretrained_backbone: bool = False
    grad_clip_norm: float = 0.0
    noise_warmup_epochs: int = 0
    dataloader_timeout_sec: float = 0.0
    dataloader_persistent_workers: bool = False
    noise_debug_log_first_batch: bool = False


@dataclass
class CompressionConfig(SharedExperimentConfig):
    """Tiny ImageNet compression at 64×64."""

    lambda_rd: float = 0.0130
    compressai_quality: int = 4
    compressai_metric: str = "mse"
    compressai_pretrained: bool = False


def _merge_dict(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {p}")
    return data


def _instantiate_dataclass(cls: Type[T], raw: Mapping[str, Any]) -> T:
    """Filter unknown keys and construct a dataclass instance."""
    r = dict(raw)
    opt_raw = r.pop("optim", None)
    optim: OptimConfig
    if isinstance(opt_raw, dict) and opt_raw:
        optim = OptimConfig(**opt_raw)
    else:
        optim = OptimConfig()
    names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in r.items() if k in names}
    return cls(optim=optim, **filtered)  # type: ignore[arg-type,misc]


def build_classification_config(raw: Mapping[str, Any]) -> ClassificationConfig:
    """Instantiate :class:`ClassificationConfig` from a nested dict."""
    return _instantiate_dataclass(ClassificationConfig, raw)


def build_compression_config(raw: Mapping[str, Any]) -> CompressionConfig:
    """Instantiate :class:`CompressionConfig` from a nested dict."""
    return _instantiate_dataclass(CompressionConfig, raw)


def load_classification_config(path: Path | str) -> ClassificationConfig:
    """Load YAML file into :class:`ClassificationConfig`."""
    data = load_yaml(path)
    return build_classification_config(data)


def load_compression_config(path: Path | str) -> CompressionConfig:
    """Load YAML file into :class:`CompressionConfig`."""
    data = load_yaml(path)
    return build_compression_config(data)
