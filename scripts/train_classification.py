"""Train a Tiny ImageNet classification model (clean or noisy uniform weight noise)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _bootstrap import ensure_package_on_path

REPO = ensure_package_on_path()

from weight_noise_ptq.classification.train import train_classification  # noqa: E402
from weight_noise_ptq.common.config import load_classification_config  # noqa: E402
from weight_noise_ptq.common.logging_setup import configure_experiment_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Path to classification YAML config.")
    p.add_argument("--seed", type=int, required=True, help="Run seed (e.g. 0, 1, 2).")
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override Tiny ImageNet root (default: config data_root).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g. cuda:0, cpu). Default: cuda if available else cpu.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: config).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory containing classification/ and compression/ (default: <repo>/results).",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root for metadata (default: auto-detect from pyproject.toml).",
    )
    return p.parse_args()


def main() -> None:
    configure_experiment_logging(level=logging.INFO)
    args = parse_args()
    cfg = load_classification_config(args.config)
    train_classification(
        cfg,
        seed=args.seed,
        data_root=args.data_root,
        device=args.device,
        num_workers=args.num_workers,
        results_base=args.output_root,
        repo_root_override=args.repo_root,
    )


if __name__ == "__main__":
    main()
