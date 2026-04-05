"""TorchVision classification models with Tiny ImageNet (200-way) heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
)

from weight_noise_ptq.common.locked_names import CLASSIFICATION_MODELS as _LOCKED_MODELS

ClassificationModelName = Literal["resnet50", "mobilenetv3_large", "convnext_tiny"]


@dataclass(frozen=True)
class ClassificationModelMetadata:
    """Static description for logging and config validation."""

    name: ClassificationModelName
    torchvision_hub: str
    num_classes: int


_METADATA: dict[ClassificationModelName, ClassificationModelMetadata] = {
    "resnet50": ClassificationModelMetadata(
        name="resnet50",
        torchvision_hub="torchvision.models.resnet50",
        num_classes=200,
    ),
    "mobilenetv3_large": ClassificationModelMetadata(
        name="mobilenetv3_large",
        torchvision_hub="torchvision.models.mobilenet_v3_large",
        num_classes=200,
    ),
    "convnext_tiny": ClassificationModelMetadata(
        name="convnext_tiny",
        torchvision_hub="torchvision.models.convnext_tiny",
        num_classes=200,
    ),
}


def get_classification_metadata(name: ClassificationModelName | str) -> ClassificationModelMetadata:
    """Return canonical metadata for ``name``."""
    key = str(name)
    if key not in _METADATA:
        raise KeyError(f"Unknown classification model: {name}. Expected one of {list(_METADATA)}.")
    return _METADATA[key]  # type: ignore[index]


def build_classification_model(
    name: ClassificationModelName | str,
    *,
    num_classes: int = 200,
    pretrained: bool = False,
) -> nn.Module:
    """Construct a torchvision model with a ``num_classes``-way classifier head."""
    key = str(name)
    w = None
    if pretrained:
        if key == "resnet50":
            w = ResNet50_Weights.DEFAULT
        elif key == "mobilenetv3_large":
            w = MobileNet_V3_Large_Weights.DEFAULT
        elif key == "convnext_tiny":
            w = ConvNeXt_Tiny_Weights.DEFAULT

    if key == "resnet50":
        m = models.resnet50(weights=w)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m

    if key == "mobilenetv3_large":
        m = models.mobilenet_v3_large(weights=w)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        return m

    if key == "convnext_tiny":
        m = models.convnext_tiny(weights=w)
        in_features = m.classifier[-1].in_features  # type: ignore[union-attr]
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        return m

    raise ValueError(f"Unknown classification model: {name}")


assert frozenset(_METADATA.keys()) == frozenset(_LOCKED_MODELS), (
    "classification/registry _METADATA keys must match locked_names.CLASSIFICATION_MODELS"
)
