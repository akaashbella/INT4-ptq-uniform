"""Eval JSON writers and CSV export helpers (locked column schema).

**Export contract (thesis runs):**

- Per-run JSON: ``eval_fp32.json`` and ``eval_quant.json`` use a ``{"rows": [...]}`` wrapper
  (or a single dict / list, parsed by :func:`rebuild_master_csvs_from_results_tree`).
- Each row must include a correct ``task`` (``classification`` | ``compression``). Rows are
  rejected if ``task`` disagrees with the directory walk (``classification/`` vs
  ``compression/``) to prevent cross-task metric contamination in master CSVs.
- Master CSV columns are fixed in :data:`MASTER_RESULTS_COLUMNS`; classification rows use
  ``top1``/``loss``/``drop_from_fp32``; compression rows use ``psnr``/``bpp``/``rd_loss`` and
  related deltas — unused fields are empty strings in CSV cells.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from weight_noise_ptq.common.logging_utils import save_json

MASTER_RESULTS_COLUMNS: tuple[str, ...] = (
    "task",
    "dataset",
    "model",
    "regime",
    "alpha",
    "lambda_rd",
    "seed",
    "checkpoint",
    "bitwidth",
    "quant_mode",
    "noise_scale_mode",
    "epoch_best",
    "top1",
    "loss",
    "drop_from_fp32",
    "retention_ratio",
    "psnr",
    "bpp",
    "psnr_drop_from_fp32",
    "bpp_shift_from_fp32",
    "rd_loss",
    "run_dir",
)


def _blank_row() -> dict[str, str]:
    return {c: "" for c in MASTER_RESULTS_COLUMNS}


def normalize_master_row(row: Mapping[str, Any]) -> dict[str, str]:
    """Return a full row dict with every locked column present (stringified)."""
    out = _blank_row()
    for k in MASTER_RESULTS_COLUMNS:
        if k in row and row[k] is not None:
            v = row[k]
            out[k] = "" if v == "" else str(v)
    return out


def write_eval_fp32_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``eval_fp32.json`` (sorted keys for stable diffs)."""
    save_json(Path(path), dict(payload))


def write_eval_quant_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``eval_quant.json`` (typically includes per-bitwidth results)."""
    save_json(Path(path), dict(payload))


def append_master_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append a single row to ``master_results.csv`` (legacy; prefer full rebuild)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_master_row(row)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(MASTER_RESULTS_COLUMNS))
        if write_header:
            w.writeheader()
        w.writerow(normalized)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    """Write CSV with exact column order (atomic replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    tmp.replace(path)


def rebuild_master_csvs_from_results_tree(
    results_root: Path,
    *,
    classification_csv_name: str = "classification_results.csv",
    compression_csv_name: str = "compression_results.csv",
    master_csv_name: str = "master_results.csv",
    summary_name: str = "summary.json",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Scan ``results/`` for run folders and rebuild CSVs deterministically.

    Reads ``eval_fp32.json`` and ``eval_quant.json`` when present. Sorting:
    ``(task, model, regime, seed, bitwidth, checkpoint, run_dir)``. Overwrites CSVs
    (no duplicate accumulation across reruns).
    """
    results_root = Path(results_root)
    rows: list[dict[str, str]] = []

    def walk_task(task: str) -> None:
        task_dir = results_root / task
        if not task_dir.is_dir():
            return
        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for regime_dir in sorted(model_dir.iterdir()):
                if not regime_dir.is_dir():
                    continue
                for seed_dir in sorted(regime_dir.iterdir()):
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                        continue
                    rows.extend(_collect_rows_from_run_dir(task, seed_dir))

    walk_task("classification")
    walk_task("compression")

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r.get("task", ""),
            r.get("model", ""),
            r.get("regime", ""),
            int(r.get("seed", 0) or 0),
            _bitwidth_sort_key(str(r.get("bitwidth", ""))),
            r.get("checkpoint", ""),
            r.get("run_dir", ""),
        ),
    )

    cls_rows = [r for r in rows_sorted if r.get("task") == "classification"]
    cmp_rows = [r for r in rows_sorted if r.get("task") == "compression"]

    master_path = results_root / master_csv_name
    cls_path = results_root / classification_csv_name
    cmp_path = results_root / compression_csv_name

    write_csv_atomic(master_path, rows_sorted, MASTER_RESULTS_COLUMNS)
    write_csv_atomic(cls_path, cls_rows, MASTER_RESULTS_COLUMNS)
    write_csv_atomic(cmp_path, cmp_rows, MASTER_RESULTS_COLUMNS)

    summary: dict[str, Any] = {
        "results_root": str(results_root.resolve()),
        "num_rows_master": len(rows_sorted),
        "num_rows_classification": len(cls_rows),
        "num_rows_compression": len(cmp_rows),
    }
    save_json(results_root / summary_name, summary)
    return rows_sorted, summary


def _bitwidth_sort_key(b: str) -> tuple[int, str]:
    order = {"fp32": 0, "w8": 1, "w6": 2, "w4": 3}
    return (order.get(b, 99), b)


def _collect_rows_from_run_dir(task: str, run_dir: Path) -> list[dict[str, str]]:
    """Load ``eval_fp32.json`` and ``eval_quant.json`` from a single run directory."""
    out: list[dict[str, str]] = []
    run_dir_s = str(run_dir.resolve())
    eval_fp = run_dir / "eval_fp32.json"
    eval_q = run_dir / "eval_quant.json"
    if eval_fp.is_file():
        with eval_fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        out.extend(_parse_eval_payload_to_rows(data, run_dir_s, task_hint=task))
    if eval_q.is_file():
        with eval_q.open("r", encoding="utf-8") as f:
            qdata = json.load(f)
        out.extend(_parse_eval_payload_to_rows(qdata, run_dir_s, task_hint=task))
    return out


def _parse_eval_payload_to_rows(
    data: Any,
    run_dir_s: str,
    *,
    task_hint: str,
) -> list[dict[str, str]]:
    """Normalize eval JSON (object with ``rows``, single object, or list) to CSV rows."""
    out: list[dict[str, str]] = []
    if isinstance(data, dict) and "rows" in data and isinstance(data["rows"], list):
        for sub in data["rows"]:
            if isinstance(sub, dict):
                merged = _ensure_run_dir(sub, run_dir_s, task_hint=task_hint)
                out.append(normalize_master_row(merged))
        return out
    if isinstance(data, list):
        for sub in data:
            if isinstance(sub, dict):
                merged = _ensure_run_dir(sub, run_dir_s, task_hint=task_hint)
                out.append(normalize_master_row(merged))
        return out
    if isinstance(data, dict):
        merged = _ensure_run_dir(data, run_dir_s, task_hint=task_hint)
        out.append(normalize_master_row(merged))
        return out
    return out


def _ensure_run_dir(
    row: Mapping[str, Any],
    run_dir: str,
    *,
    task_hint: str = "",
) -> dict[str, Any]:
    r = dict(row)
    r.setdefault("run_dir", run_dir)
    existing = r.get("task")
    if existing and task_hint and str(existing) != str(task_hint):
        raise ValueError(
            f"Row task {existing!r} does not match results tree {task_hint!r} (run_dir={run_dir}). "
            "Refusing to export — fix eval JSON or directory layout to avoid mixing tasks."
        )
    if not existing and task_hint:
        r["task"] = task_hint
    r.setdefault("task", "")
    return r


def classification_eval_fp32_payload(
    *,
    task: str,
    dataset: str,
    model: str,
    regime: str,
    alpha: float,
    seed: int,
    checkpoint: str,
    bitwidth: str,
    quant_mode: str,
    noise_scale_mode: str,
    epoch_best: int,
    top1: float,
    loss: float,
    drop_from_fp32: str | float,
    retention_ratio: str | float,
    run_dir: str,
) -> dict[str, Any]:
    """Canonical single-row payload for classification fp32 (wrapped in ``rows``)."""
    row = {
        "task": task,
        "dataset": dataset,
        "model": model,
        "regime": regime,
        "alpha": alpha,
        "lambda_rd": "",
        "seed": seed,
        "checkpoint": checkpoint,
        "bitwidth": bitwidth,
        "quant_mode": quant_mode,
        "noise_scale_mode": noise_scale_mode,
        "epoch_best": epoch_best,
        "top1": top1,
        "loss": loss,
        "drop_from_fp32": drop_from_fp32,
        "retention_ratio": retention_ratio,
        "psnr": "",
        "bpp": "",
        "psnr_drop_from_fp32": "",
        "bpp_shift_from_fp32": "",
        "rd_loss": "",
        "run_dir": run_dir,
    }
    return {"rows": [row]}


def compression_eval_fp32_payload(
    *,
    task: str,
    dataset: str,
    model: str,
    regime: str,
    alpha: float,
    lambda_rd: float,
    seed: int,
    checkpoint: str,
    bitwidth: str,
    quant_mode: str,
    noise_scale_mode: str,
    epoch_best: int,
    psnr: float,
    bpp: float,
    psnr_drop_from_fp32: str | float,
    bpp_shift_from_fp32: str | float,
    rd_loss: float,
    run_dir: str,
) -> dict[str, Any]:
    """Canonical single-row payload for compression fp32."""
    row = {
        "task": task,
        "dataset": dataset,
        "model": model,
        "regime": regime,
        "alpha": alpha,
        "lambda_rd": lambda_rd,
        "seed": seed,
        "checkpoint": checkpoint,
        "bitwidth": bitwidth,
        "quant_mode": quant_mode,
        "noise_scale_mode": noise_scale_mode,
        "epoch_best": epoch_best,
        "top1": "",
        "loss": "",
        "drop_from_fp32": "",
        "retention_ratio": "",
        "psnr": psnr,
        "bpp": bpp,
        "psnr_drop_from_fp32": psnr_drop_from_fp32,
        "bpp_shift_from_fp32": bpp_shift_from_fp32,
        "rd_loss": rd_loss,
        "run_dir": run_dir,
    }
    return {"rows": [row]}


def quant_eval_payload(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Wrap multiple quantized rows for ``eval_quant.json``."""
    return {"rows": [normalize_master_row(r) for r in rows]}
