"""Shared Tiny ImageNet path layout, val annotations, and class indexing (single source of truth)."""

from __future__ import annotations

from pathlib import Path

# Locked dataset: 200 classes (WordNet ids) under train/<wnid>/images/
TINY_IMAGENET_NUM_CLASSES: int = 200


def build_wnid_to_class_idx(train_root: Path) -> dict[str, int]:
    """Map WordNet id (folder name under ``train/``) to a stable contiguous label ``0..199``.

    Ordering is lexicographic on wnid so train and val use the same mapping.
    Raises if the number of class folders is not exactly 200 (catches bad trees).
    """
    if not train_root.is_dir():
        raise FileNotFoundError(f"Tiny ImageNet train root not found: {train_root}")
    wnids = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
    if len(wnids) != TINY_IMAGENET_NUM_CLASSES:
        raise ValueError(
            f"Expected {TINY_IMAGENET_NUM_CLASSES} class folders under {train_root}, found {len(wnids)}",
        )
    return {w: i for i, w in enumerate(wnids)}


def read_val_annotations(val_root: Path) -> dict[str, str]:
    """Parse ``val_annotations.txt`` -> ``{image_filename -> wnid}``.

    Standard Tiny ImageNet format: ``<image_name> <wnid> [<bbox>...]`` (tab or
    space separated); only the first two fields are used (bbox is ignored).

    Non-empty lines with fewer than two fields are treated as corruption and
    raise — silent skips would hide a broken download and skew val integrity.
    """
    ann_path = val_root / "val_annotations.txt"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing val_annotations.txt at {ann_path}")
    mapping: dict[str, str] = {}
    with ann_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"{ann_path}: line {lineno}: expected filename and wnid, got {line!r}",
                )
            filename, wnid = parts[0], parts[1]
            mapping[filename] = wnid
    return mapping
