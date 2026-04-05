"""Print or save reproducibility metadata (no training required).

Uses :func:`weight_noise_ptq.common.environment.collect_environment_metadata`.
Safe to run on login nodes or laptops without a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import ensure_package_on_path

ensure_package_on_path()

from weight_noise_ptq.common.environment import collect_environment_metadata  # noqa: E402
from weight_noise_ptq.common.logging_utils import save_json  # noqa: E402
from weight_noise_ptq.common.paths import repo_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON to this path (atomic). If omitted, print JSON to stdout only.",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Git / project root for commit detection (default: auto-detect).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written but do not save a file (stdout still shows JSON).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root if args.repo_root is not None else repo_root()
    meta = collect_environment_metadata(repo_root_path=str(root))
    text = json.dumps(meta, indent=2, sort_keys=True)
    print(text)
    if args.output and not args.dry_run:
        save_json(args.output, meta)
        print(f"[report_environment] Wrote {args.output}", file=sys.stderr)
    elif args.output and args.dry_run:
        print(f"[report_environment] Dry-run: would write {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
