"""Rebuild master and per-task CSVs from ``eval_*.json`` files under ``results/``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _bootstrap import ensure_package_on_path

ensure_package_on_path()

from weight_noise_ptq.common.logging_setup import configure_experiment_logging  # noqa: E402
from weight_noise_ptq.common.results_export import rebuild_master_csvs_from_results_tree  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Path to the results directory (contains classification/, compression/, and exported CSVs).",
    )
    return p.parse_args()


def main() -> None:
    configure_experiment_logging(level=logging.INFO)
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.results_root if args.results_root is not None else repo / "results"
    rows, summary = rebuild_master_csvs_from_results_tree(root)
    logging.info("Wrote CSVs under %s (%s master rows).", root.resolve(), len(rows))
    logging.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
