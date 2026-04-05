"""CSV append-only log writers and JSON helpers for experiment I/O."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

_TRAIN_COLUMNS: tuple[str, ...] = (
    "epoch",
    "step",
    "lr",
    "loss",
    "regime",
)
_VAL_COLUMNS: tuple[str, ...] = (
    "epoch",
    "metric_name",
    "metric_value",
    "regime",
)


def train_log_columns() -> tuple[str, ...]:
    """Column order for ``train_log.csv``."""
    return _TRAIN_COLUMNS


def val_log_columns() -> tuple[str, ...]:
    """Column order for ``val_log.csv``."""
    return _VAL_COLUMNS


class CsvAppendWriter:
    """Append rows to a CSV file, writing the header once if missing."""

    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = Path(path)
        self.fieldnames: tuple[str, ...] = tuple(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_header_if_needed()

    def _write_header_if_needed(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        with self.path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writeheader()

    def writerow(self, row: Mapping[str, Any]) -> None:
        """Append one row; keys must match ``fieldnames``."""
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            w.writerow({k: row.get(k, "") for k in self.fieldnames})

    def writerows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for r in rows:
            self.writerow(r)


class TrainLogWriter(CsvAppendWriter):
    """Training step log compatible with :func:`train_log_columns`."""

    def __init__(self, path: Path) -> None:
        super().__init__(path, train_log_columns())


class ValLogWriter(CsvAppendWriter):
    """Validation epoch log compatible with :func:`val_log_columns`."""

    def __init__(self, path: Path) -> None:
        super().__init__(path, val_log_columns())


def save_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """Write ``data`` as UTF-8 JSON (atomic replace via temp file).

    Uses ``allow_nan=False`` so NaN/Inf cannot silently serialize to invalid JSON
    or platform-dependent ``null`` behavior — thesis runs should surface bad floats.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=indent,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
    except (TypeError, ValueError) as e:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise ValueError(
            f"Cannot serialize data to JSON for {path} (non-finite floats, NaN, or non-JSON types): {e}"
        ) from e
    tmp.replace(path)


def load_json(path: Path) -> Any:
    """Load JSON from ``path``."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_json(
    base: MutableMapping[str, Any],
    updates: Mapping[str, Any],
    *,
    deep: bool = False,
) -> MutableMapping[str, Any]:
    """Merge ``updates`` into ``base`` (shallow by default)."""
    for k, v in updates.items():
        if (
            deep
            and k in base
            and isinstance(base[k], dict)
            and isinstance(v, dict)
        ):
            merge_json(base[k], v, deep=True)  # type: ignore[arg-type]
        else:
            base[k] = v  # type: ignore[index]
    return base
