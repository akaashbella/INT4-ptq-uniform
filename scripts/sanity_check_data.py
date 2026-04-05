"""Verify Tiny ImageNet directory layout and dataloaders (one batch each).

Usage:

  python scripts/sanity_check_data.py --data-root /path/to/tiny-imagenet-200

Requires the full dataset with train/val and val_annotations.txt (see README).
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import ensure_package_on_path

ensure_package_on_path()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from weight_noise_ptq.classification.datasets import (  # noqa: E402
    TinyImageNetClassificationDataset,
    train_transforms_224,
    val_transforms_224,
)
from weight_noise_ptq.compression.datasets import (  # noqa: E402
    TinyImageNetCompressionDataset,
    train_transforms_64,
    val_transforms_64,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, required=True, help="Tiny ImageNet root directory.")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 for debug).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.data_root

    cls_train = TinyImageNetClassificationDataset(root, split="train", transform=train_transforms_224())
    cls_val = TinyImageNetClassificationDataset(root, split="val", transform=val_transforms_224())
    cmp_train = TinyImageNetCompressionDataset(root, split="train", transform=train_transforms_64())
    cmp_val = TinyImageNetCompressionDataset(root, split="val", transform=val_transforms_64())

    for name, ds in (
        ("cls_train", cls_train),
        ("cls_val", cls_val),
        ("cmp_train", cmp_train),
        ("cmp_val", cmp_val),
    ):
        print(f"{name}: n={len(ds)}")

    dl_cls = DataLoader(cls_train, batch_size=4, shuffle=False, num_workers=args.num_workers)
    dl_cmp = DataLoader(cmp_train, batch_size=4, shuffle=False, num_workers=args.num_workers)

    xb, yb = next(iter(dl_cls))
    assert xb.shape[1:] == (3, 224, 224) and yb.shape == (4,)
    xc = next(iter(dl_cmp))
    assert xc.shape[1:] == (3, 64, 64)

    print("sanity_check_data: loaders OK.", file=sys.stderr)


if __name__ == "__main__":
    main()
