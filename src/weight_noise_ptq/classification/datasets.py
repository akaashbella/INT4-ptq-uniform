"""Tiny ImageNet classification: 224×224 ImageNet-style preprocessing and correct val labels."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from weight_noise_ptq.common.tiny_imagenet_io import (
    build_wnid_to_class_idx,
    read_val_annotations,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TinyImageNetClassificationDataset(Dataset[tuple[torch.Tensor, int]]):
    """Tiny ImageNet for classification with correct validation label mapping.

    Train images live under ``<root>/train/<wnid>/images/*.JPEG``.
    Val images live under ``<root>/val/images`` with labels from ``val_annotations.txt``.
    Class indices match :func:`weight_noise_ptq.common.tiny_imagenet_io.build_wnid_to_class_idx`.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform

        train_root = self.root / "train"
        self._wnid_to_idx = build_wnid_to_class_idx(train_root)
        self._samples: list[tuple[Path, int]] = []

        if split == "train":
            for wnid, idx in self._wnid_to_idx.items():
                class_dir = train_root / wnid / "images"
                if not class_dir.is_dir():
                    continue
                seen: set[str] = set()
                for p in sorted(class_dir.glob("*.JPEG")) + sorted(class_dir.glob("*.jpg")):
                    key = str(p.resolve())
                    if key not in seen:
                        seen.add(key)
                        self._samples.append((p, idx))
        elif split == "val":
            val_root = self.root / "val"
            images_dir = val_root / "images"
            ann = read_val_annotations(val_root)
            if not images_dir.is_dir():
                raise FileNotFoundError(f"Val images dir missing: {images_dir}")
            missing: list[str] = []
            for fname, wnid in sorted(ann.items()):
                path = images_dir / fname
                if not path.is_file():
                    missing.append(fname)
                    continue
                if wnid not in self._wnid_to_idx:
                    raise KeyError(f"Unknown wnid in val_annotations: {wnid} for {fname}")
                self._samples.append((path, self._wnid_to_idx[wnid]))
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} val images listed in val_annotations.txt are missing under {images_dir} "
                    f"(e.g. {missing[:3]})",
                )
        else:
            raise ValueError("split must be 'train' or 'val'")

        if not self._samples:
            raise RuntimeError(f"No samples found for split={split} under {self.root}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, y = self._samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(img)
        else:
            x = transforms.ToTensor()(img)
        return x, y


def train_transforms_224() -> transforms.Compose:
    """Standard training augmentation + 224×224 + ImageNet normalization."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ],
    )


def val_transforms_224() -> transforms.Compose:
    """224×224 center crop + ImageNet normalization (evaluation)."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ],
    )
