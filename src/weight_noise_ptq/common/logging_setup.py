"""Stdout-oriented logging for batch / HPC jobs (no TTY assumptions)."""

from __future__ import annotations

import logging
import sys
from typing import TextIO


def configure_experiment_logging(
    *,
    level: int = logging.INFO,
    stream: TextIO | None = None,
    force: bool = True,
) -> None:
    """Configure root logging for non-interactive runs (Slurm, PBS, etc.).

    Uses a single stream (default ``sys.stdout``), ISO-like timestamps, and
    forces reconfiguration if the process already configured logging (``force``
    matches Python 3.8+ :func:`logging.basicConfig` behavior).
    """
    if stream is None:
        stream = sys.stdout
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    try:
        logging.basicConfig(
            level=level,
            format=fmt,
            datefmt=datefmt,
            stream=stream,
            force=force,
        )
    except TypeError:
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=stream)
