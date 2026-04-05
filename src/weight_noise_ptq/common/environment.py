"""Capture run environment metadata for reproducibility (``environment.json``)."""

from __future__ import annotations

import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import torch


def _git_rev_parse(repo_root: str | None, *, short: bool) -> str | None:
    """Return git SHA (full or short) if available."""
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        kwargs: dict[str, Any] = {"cwd": repo_root} if repo_root else {}
        out = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            **kwargs,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _git_commit_short(repo_root: str | None = None) -> str | None:
    """Return abbreviated git SHA if available, else ``None``."""
    return _git_rev_parse(repo_root, short=True)


def git_commit_full(repo_root: str | None = None) -> str | None:
    """Return full git SHA if available, else ``None``."""
    return _git_rev_parse(repo_root, short=False)


def collect_environment_metadata(*, repo_root_path: str | None = None) -> dict[str, Any]:
    """Collect Python, library versions, device, OS, hostname, and UTC timestamp.

    Intended to be written once per run as ``environment.json`` alongside ``config.json``.
    """
    try:
        import torchvision  # noqa: WPS433 — runtime optional pattern
        tv_ver = torchvision.__version__
    except Exception:
        tv_ver = None

    try:
        import compressai  # noqa: WPS433
        ca_ver = compressai.__version__
    except Exception:
        ca_ver = None

    try:
        import numpy as np  # noqa: WPS433
        numpy_ver = np.__version__
    except Exception:
        numpy_ver = None

    cuda_available = torch.cuda.is_available()
    cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_str = "cuda" if cuda_available else "cpu"
    if cuda_available:
        try:
            device_str = torch.cuda.get_device_name(0)
        except Exception:
            device_str = "cuda"

    now = datetime.now(timezone.utc).isoformat()

    meta: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "os": sys.platform,
        "hostname": socket.gethostname(),
        "run_start_timestamp_utc": now,
        "torch": torch.__version__,
        "torchvision": tv_ver,
        "compressai": ca_ver,
        "numpy": numpy_ver,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_count,
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_description": device_str,
        "git_commit_short": _git_commit_short(repo_root_path),
        "git_commit_full": git_commit_full(repo_root_path),
    }
    # Backward-compatible key used in earlier runs
    meta["git_commit"] = meta["git_commit_short"]
    return meta
