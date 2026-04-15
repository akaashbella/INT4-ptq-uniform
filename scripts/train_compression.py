"""Train a CompressAI model on Tiny ImageNet 64×64 (MSE rate–distortion, λ=0.0130)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _bootstrap import ensure_package_on_path

REPO = ensure_package_on_path()

from weight_noise_ptq.common.config import load_compression_config  # noqa: E402
from weight_noise_ptq.common.logging_setup import configure_experiment_logging  # noqa: E402
from weight_noise_ptq.compression.train import train_compression  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Path to compression YAML config.")
    p.add_argument("--seed", type=int, required=True, help="Run seed.")
    p.add_argument("--data-root", type=str, default=None, help="Override dataset root.")
    p.add_argument("--device", type=str, default=None, help="Torch device (default: auto).")
    p.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Results directory containing classification/ and compression/.",
    )
    p.add_argument("--repo-root", type=Path, default=None, help="Repo root for environment.json.")
    return p.parse_args()


def main() -> None:
    configure_experiment_logging(level=logging.INFO)
    args = parse_args()
    cfg = load_compression_config(args.config)
    resolved_data_root = Path(args.data_root) if args.data_root is not None else Path(cfg.data_root)
    if (not str(resolved_data_root).strip()) or "/path/to/" in str(resolved_data_root) or (not resolved_data_root.is_dir()):
        raise ValueError(f"Invalid data_root for Tiny ImageNet: {resolved_data_root}")
    logger.info(
        "Launch config=%s model=%s regime=%s seed=%s device=%s data_root=%s",
        args.config,
        cfg.model,
        cfg.regime,
        args.seed,
        args.device or "auto",
        resolved_data_root,
    )
    try:
        train_compression(
            cfg,
            seed=args.seed,
            data_root=resolved_data_root,
            device=args.device,
            num_workers=args.num_workers,
            results_base=args.output_root,
            repo_root_override=args.repo_root,
        )
    except Exception:
        logger.exception(
            "Compression training failed: config=%s model=%s regime=%s seed=%s data_root=%s",
            args.config,
            cfg.model,
            cfg.regime,
            args.seed,
            resolved_data_root,
        )
        raise


if __name__ == "__main__":
    main()
