"""Tiny ImageNet for image compression: 64×64 floats in ``[0, 1]`` (reconstruction, no labels)."""

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


class TinyImageNetCompressionDataset(Dataset[torch.Tensor]):
    """Tiny ImageNet images resized to 64×64 for learned compression.

    Returns ``x`` with shape ``[3, 64, 64]`` in ``[0, 1]``. Train/val image lists align
    with classification (same files) but labels are not used.
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
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.split = split
        self.transform = transform

        train_root = self.root / "train"
        wnid_to_idx = build_wnid_to_class_idx(train_root)
        self._paths: list[Path] = []

        if split == "train":
            for wnid in sorted(wnid_to_idx.keys()):
                class_dir = train_root / wnid / "images"
                if not class_dir.is_dir():
                    continue
                seen: set[str] = set()
                for p in sorted(class_dir.glob("*.JPEG")) + sorted(class_dir.glob("*.jpg")):
                    key = str(p.resolve())
                    if key not in seen:
                        seen.add(key)
                        self._paths.append(p)
        else:
            val_root = self.root / "val"
            images_dir = val_root / "images"
            ann = read_val_annotations(val_root)
            if not images_dir.is_dir():
                raise FileNotFoundError(f"Val images dir missing: {images_dir}")
            missing: list[str] = []
            for fname in sorted(ann.keys()):
                wnid = ann[fname]
                if wnid not in wnid_to_idx:
                    raise KeyError(f"Val wnid {wnid!r} for {fname!r} not in train class set")
                path = images_dir / fname
                if not path.is_file():
                    missing.append(fname)
                    continue
                self._paths.append(path)
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} val images listed in val_annotations.txt are missing under {images_dir} "
                    f"(e.g. {missing[:3]})",
                )

        if not self._paths:
            raise RuntimeError(f"No images for split={split} under {self.root}")

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self._paths[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            return self.transform(img)
        return transforms.ToTensor()(img)


def train_transforms_64() -> transforms.Compose:
    """Augmentation + 64×64 reconstruction inputs in ``[0,1]``."""
    return transforms.Compose(
        [
            transforms.RandomCrop(64, pad_if_needed=True, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ],
    )


def val_transforms_64() -> transforms.Compose:
    """Deterministic 64×64 evaluation pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ],
    )
