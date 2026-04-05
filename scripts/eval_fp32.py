"""FP32 evaluation for a trained run (writes ``eval_fp32.json``)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _bootstrap import ensure_package_on_path

ensure_package_on_path()

from weight_noise_ptq.common.config import (  # noqa: E402
    load_classification_config,
    load_compression_config,
    load_yaml,
)
from weight_noise_ptq.common.logging_setup import configure_experiment_logging  # noqa: E402
from weight_noise_ptq.common.validators import (  # noqa: E402
    validate_classification_config,
    validate_compression_config,
)
from weight_noise_ptq.eval_runs import (  # noqa: E402
    run_eval_fp32_classification,
    run_eval_fp32_compression,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="YAML config for this model/regime.")
    p.add_argument("--seed", type=int, required=True, help="Run seed index.")
    p.add_argument("--checkpoint", choices=("best", "last"), default="best", help="Checkpoint stem.")
    p.add_argument("--data-root", type=str, default=None, help="Override Tiny ImageNet root.")
    p.add_argument("--device", type=str, default=None, help="Torch device.")
    p.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    p.add_argument("--output-root", type=Path, default=None, help="Results root (classification/… parent).")
    return p.parse_args()


def main() -> None:
    configure_experiment_logging(level=logging.INFO)
    args = parse_args()
    raw = load_yaml(args.config)
    task = str(raw.get("task", ""))

    if task == "classification":
        cfg = load_classification_config(args.config)
        validate_classification_config(cfg)
        run_eval_fp32_classification(
            cfg,
            seed=args.seed,
            checkpoint_name=args.checkpoint,
            data_root=args.data_root,
            device=args.device,
            num_workers=args.num_workers,
            results_base=args.output_root,
        )
    elif task == "compression":
        cfg = load_compression_config(args.config)
        validate_compression_config(cfg)
        run_eval_fp32_compression(
            cfg,
            seed=args.seed,
            checkpoint_name=args.checkpoint,
            data_root=args.data_root,
            device=args.device,
            num_workers=args.num_workers,
            results_base=args.output_root,
        )
    else:
        raise ValueError(f"Unknown task: {task!r}")


if __name__ == "__main__":
    main()
