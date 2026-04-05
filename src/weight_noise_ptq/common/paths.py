"""Canonical filesystem paths for the repository and experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

TaskName = Literal["classification", "compression"]

ClassificationModelName = Literal["resnet50", "mobilenetv3_large", "convnext_tiny"]
CompressionModelName = Literal["factorized_prior", "scale_hyperprior", "cheng2020_attention"]
RegimeName = Literal["clean", "noisy_uniform_a0.02"]


def repo_root() -> Path:
    """Return the repository root (directory containing ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate repository root (pyproject.toml not found).")


def results_root(explicit_results_dir: Path | None = None) -> Path:
    """Return the directory that contains ``classification/`` and ``compression/``.

    If ``explicit_results_dir`` is set, it must be that directory (typically
    ``<repo>/results``). If ``None``, defaults to ``<repo>/results``.
    """
    if explicit_results_dir is not None:
        return Path(explicit_results_dir)
    return repo_root() / "results"


def classification_run_dir(
    model: ClassificationModelName | str,
    regime: RegimeName | str,
    seed: int,
    *,
    results_base: Path | None = None,
) -> Path:
    """``<results>/classification/<model>/<regime>/seed_<seed>/``."""
    return results_root(results_base) / "classification" / str(model) / str(regime) / f"seed_{int(seed)}"


def compression_run_dir(
    model: CompressionModelName | str,
    regime: RegimeName | str,
    seed: int,
    *,
    results_base: Path | None = None,
) -> Path:
    """``<results>/compression/<model>/<regime>/seed_<seed>/``."""
    return results_root(results_base) / "compression" / str(model) / str(regime) / f"seed_{int(seed)}"


def run_dir_for_task(
    task: TaskName | str,
    model: str,
    regime: str,
    seed: int,
    *,
    results_base: Path | None = None,
) -> Path:
    """Dispatch to :func:`classification_run_dir` or :func:`compression_run_dir`."""
    if task == "classification":
        return classification_run_dir(model, regime, seed, results_base=results_base)
    if task == "compression":
        return compression_run_dir(model, regime, seed, results_base=results_base)
    raise ValueError(f"Unknown task: {task}")
