"""Ensure ``src/`` is on ``sys.path`` when running scripts without editable install."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_package_on_path() -> Path:
    """Insert ``<repo>/src`` into ``sys.path`` and return repository root."""
    repo = Path(__file__).resolve().parents[1]
    src = repo / "src"
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
    return repo
