"""Generate bash commands for the locked experiment matrix (does not execute them).

Prints one command per line to stdout. Use ``--output script.sh`` to save a shell
script; use ``--dry-run`` to print only and never write a file.

Intended for HPC: paste into Slurm job arrays, ``parallel``, or manual submission.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import ensure_package_on_path

ensure_package_on_path()

from weight_noise_ptq.common.locked_names import (  # noqa: E402
    CLASSIFICATION_MODELS,
    COMPRESSION_MODELS,
    REGIMES,
    RUN_SEEDS,
)
from weight_noise_ptq.common.paths import repo_root  # noqa: E402


def _p(path: Path) -> str:
    """POSIX path for bash scripts (portable when generating on Windows)."""
    return path.as_posix()


SEEDS = RUN_SEEDS


def _emit_train_classification(
    repo: Path,
    *,
    data_root_var: str,
    output_root_var: str,
    python_var: str,
) -> list[str]:
    lines: list[str] = []
    base = repo / "configs" / "classification"
    script = _p(repo / "scripts" / "train_classification.py")
    for model in CLASSIFICATION_MODELS:
        for regime in REGIMES:
            cfg = base / f"{model}_{regime}.yaml"
            for seed in SEEDS:
                lines.append(
                    f'{python_var} "{script}" '
                    f'--config "{_p(cfg)}" --seed {seed} '
                    f'--data-root "{data_root_var}" --output-root "{output_root_var}"'
                )
    return lines


def _emit_train_compression(
    repo: Path,
    *,
    data_root_var: str,
    output_root_var: str,
    python_var: str,
) -> list[str]:
    lines: list[str] = []
    base = repo / "configs" / "compression"
    script = _p(repo / "scripts" / "train_compression.py")
    for model in COMPRESSION_MODELS:
        for regime in REGIMES:
            cfg = base / f"{model}_{regime}.yaml"
            for seed in SEEDS:
                lines.append(
                    f'{python_var} "{script}" '
                    f'--config "{_p(cfg)}" --seed {seed} '
                    f'--data-root "{data_root_var}" --output-root "{output_root_var}"'
                )
    return lines


def _emit_eval_fp32(
    repo: Path,
    *,
    data_root_var: str,
    output_root_var: str,
    python_var: str,
    checkpoint: str,
) -> list[str]:
    lines: list[str] = []
    script = _p(repo / "scripts" / "eval_fp32.py")
    for base, models in (
        (repo / "configs" / "classification", CLASSIFICATION_MODELS),
        (repo / "configs" / "compression", COMPRESSION_MODELS),
    ):
        for model in models:
            for regime in REGIMES:
                cfg = base / f"{model}_{regime}.yaml"
                for seed in SEEDS:
                    lines.append(
                        f'{python_var} "{script}" '
                        f'--config "{_p(cfg)}" --seed {seed} --checkpoint {checkpoint} '
                        f'--data-root "{data_root_var}" --output-root "{output_root_var}"'
                    )
    return lines


def _emit_eval_quant(
    repo: Path,
    *,
    data_root_var: str,
    output_root_var: str,
    python_var: str,
    checkpoint: str,
) -> list[str]:
    lines: list[str] = []
    script = _p(repo / "scripts" / "eval_quant.py")
    for base, models in (
        (repo / "configs" / "classification", CLASSIFICATION_MODELS),
        (repo / "configs" / "compression", COMPRESSION_MODELS),
    ):
        for model in models:
            for regime in REGIMES:
                cfg = base / f"{model}_{regime}.yaml"
                for seed in SEEDS:
                    lines.append(
                        f'{python_var} "{script}" '
                        f'--config "{_p(cfg)}" --seed {seed} --checkpoint {checkpoint} '
                        f'--data-root "{data_root_var}" --output-root "{output_root_var}"'
                    )
    return lines


def _emit_export_csv(repo: Path, *, output_root_var: str, python_var: str) -> list[str]:
    script = _p(repo / "scripts" / "export_master_csv.py")
    return [f'{python_var} "{script}" ' f'--results-root "{output_root_var}"']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "mode",
        choices=(
            "train-classification",
            "train-compression",
            "train-all",
            "eval-fp32",
            "eval-quant",
            "eval-all",
            "export-csv",
        ),
        help="Which command family to emit.",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: auto-detect).",
    )
    p.add_argument(
        "--data-root-placeholder",
        type=str,
        default="${DATA_ROOT}",
        help="Literal string embedded in emitted commands (default: ${DATA_ROOT}).",
    )
    p.add_argument(
        "--output-root-placeholder",
        type=str,
        default="${RESULTS_ROOT}",
        help="Literal string for --output-root / --results-root (default: ${RESULTS_ROOT}).",
    )
    p.add_argument(
        "--python-placeholder",
        type=str,
        default="${PYTHON:-python}",
        help="Python executable fragment in emitted commands.",
    )
    p.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="For eval modes: checkpoint stem (default: best).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, write commands to this file (bash).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write --output; only print commands (and stderr notice).",
    )
    p.add_argument(
        "--header",
        action="store_true",
        help="Emit a short bash header (set -euo pipefail, REPO_ROOT).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root).resolve() if args.repo_root is not None else repo_root()
    dr = args.data_root_placeholder
    out_r = args.output_root_placeholder
    py = args.python_placeholder

    lines: list[str] = []
    if args.header:
        lines.append("#!/usr/bin/env bash")
        lines.append("set -euo pipefail")
        lines.append(f'REPO_ROOT="{_p(repo)}"')
        lines.append('cd "$REPO_ROOT"')
        lines.append("")

    if args.mode == "train-classification":
        lines.extend(_emit_train_classification(repo, data_root_var=dr, output_root_var=out_r, python_var=py))
    elif args.mode == "train-compression":
        lines.extend(_emit_train_compression(repo, data_root_var=dr, output_root_var=out_r, python_var=py))
    elif args.mode == "train-all":
        lines.extend(_emit_train_classification(repo, data_root_var=dr, output_root_var=out_r, python_var=py))
        lines.extend(_emit_train_compression(repo, data_root_var=dr, output_root_var=out_r, python_var=py))
    elif args.mode == "eval-fp32":
        lines.extend(
            _emit_eval_fp32(
                repo,
                data_root_var=dr,
                output_root_var=out_r,
                python_var=py,
                checkpoint=args.checkpoint,
            )
        )
    elif args.mode == "eval-quant":
        lines.extend(
            _emit_eval_quant(
                repo,
                data_root_var=dr,
                output_root_var=out_r,
                python_var=py,
                checkpoint=args.checkpoint,
            )
        )
    elif args.mode == "eval-all":
        lines.extend(
            _emit_eval_fp32(
                repo,
                data_root_var=dr,
                output_root_var=out_r,
                python_var=py,
                checkpoint=args.checkpoint,
            )
        )
        lines.extend(
            _emit_eval_quant(
                repo,
                data_root_var=dr,
                output_root_var=out_r,
                python_var=py,
                checkpoint=args.checkpoint,
            )
        )
    elif args.mode == "export-csv":
        lines.extend(_emit_export_csv(repo, output_root_var=out_r, python_var=py))

    text = "\n".join(lines) + ("\n" if lines else "")
    print(text, end="")

    if args.output and not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[launch_commands] Wrote {args.output} ({len(lines)} lines)", file=sys.stderr)
    elif args.output and args.dry_run:
        print(f"[launch_commands] Dry-run: would write {args.output} ({len(lines)} lines)", file=sys.stderr)


if __name__ == "__main__":
    main()
