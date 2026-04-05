"""CompressAI image compression models (factorized / hyperprior / Cheng attention)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch.nn as nn
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, cheng2020_attn

from weight_noise_ptq.common.locked_names import COMPRESSION_MODELS as _LOCKED_MODELS

CompressionModelName = Literal["factorized_prior", "scale_hyperprior", "cheng2020_attention"]


@dataclass(frozen=True)
class CompressionModelMetadata:
    """Static description for logging."""

    name: CompressionModelName
    compressai_zoo: str
    default_quality: int


_METADATA: dict[CompressionModelName, CompressionModelMetadata] = {
    "factorized_prior": CompressionModelMetadata(
        name="factorized_prior",
        compressai_zoo="compressai.zoo.bmshj2018_factorized",
        default_quality=4,
    ),
    "scale_hyperprior": CompressionModelMetadata(
        name="scale_hyperprior",
        compressai_zoo="compressai.zoo.bmshj2018_hyperprior",
        default_quality=4,
    ),
    "cheng2020_attention": CompressionModelMetadata(
        name="cheng2020_attention",
        compressai_zoo="compressai.zoo.cheng2020_attn",
        default_quality=4,
    ),
}


def get_compression_metadata(name: CompressionModelName | str) -> CompressionModelMetadata:
    """Return canonical metadata for ``name``."""
    key = str(name)
    if key not in _METADATA:
        raise KeyError(f"Unknown compression model: {name}. Expected one of {list(_METADATA)}.")
    return _METADATA[key]  # type: ignore[index]


def build_compression_model(
    name: CompressionModelName | str,
    *,
    quality: int | None = None,
    metric: str = "mse",
    pretrained: bool = False,
) -> nn.Module:
    """Instantiate a CompressAI model (training from scratch unless ``pretrained``).

    Quality maps to internal channel widths per CompressAI zoo (1–8 for factorized
    and hyperprior; 1–6 for Cheng attention).
    """
    key = str(name)
    meta = get_compression_metadata(key)
    q = int(quality) if quality is not None else meta.default_quality

    if key == "factorized_prior":
        return bmshj2018_factorized(q, metric=metric, pretrained=pretrained)

    if key == "scale_hyperprior":
        return bmshj2018_hyperprior(q, metric=metric, pretrained=pretrained)

    if key == "cheng2020_attention":
        return cheng2020_attn(q, metric=metric, pretrained=pretrained)

    raise ValueError(f"Unknown compression model: {name}")


assert frozenset(_METADATA.keys()) == frozenset(_LOCKED_MODELS), (
    "compression/registry _METADATA keys must match locked_names.COMPRESSION_MODELS"
)
