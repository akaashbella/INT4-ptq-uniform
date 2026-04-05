"""Scaled uniform weight noise on eligible conv/linear weights only (temporary perturbation)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Iterator

import torch
from torch import nn


def is_eligible_weight_param(module: nn.Module, param_name: str) -> bool:
    """Return True only for ``.weight`` of ``Conv{1,2,3}d`` and ``Linear``."""
    if param_name != "weight":
        return False
    return isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear))


def iter_eligible_weight_parameters(
    model: nn.Module,
) -> Iterator[tuple[str, nn.Parameter, nn.Module]]:
    """Yield ``(qualified_name, parameter, module)`` for noise/quant eligibility.

    Only ``Conv{1,2,3}d`` and ``Linear`` *weight* tensors are included; biases,
    BN scale/shift, and other params are skipped so training noise and PTQ stay
    strictly on convolutional / linear weights.
    """
    for name, module in model.named_modules():
        if not is_eligible_weight_param(module, "weight"):
            continue
        w = module.weight
        assert isinstance(w, nn.Parameter)
        qual = f"{name}.weight" if name else "weight"
        yield qual, w, module


@dataclass
class _NoiseState:
    """Holds frozen copies for restore after perturbation."""

    tensors: list[tuple[nn.Parameter, torch.Tensor]]

    @classmethod
    def capture(cls, model: nn.Module) -> "_NoiseState":
        snaps: list[tuple[nn.Parameter, torch.Tensor]] = []
        for _, param, _ in iter_eligible_weight_parameters(model):
            snaps.append((param, param.data.detach().clone()))
        return cls(tensors=snaps)

    def restore(self) -> None:
        for param, clean in self.tensors:
            param.data.copy_(clean)


def add_uniform_noise_to_eligible_weights(
    model: nn.Module,
    *,
    alpha: float,
    generator: torch.Generator | None = None,
) -> None:
    """In-place add scaled uniform noise to eligible weights (must restore externally).

    For each tensor ``W``: ``sigma = std(W)`` (full tensor), sample
    ``eps ~ U(-alpha*sigma, +alpha*sigma)`` with the same shape as ``W``,
    then ``W <- W + eps``.

    Note: :meth:`torch.Tensor.uniform_` follows PyTorch's half-open interval
    convention on the upper end; for continuous weights this matches the
    locked design up to floating-point granularity.
    """
    gen = generator
    for _, param, _ in iter_eligible_weight_parameters(model):
        w = param.data
        sigma = torch.std(w.float(), unbiased=False).clamp_min(1e-12)
        low = -float(alpha) * sigma
        high = float(alpha) * sigma
        noise = torch.empty_like(w)
        noise.uniform_(low, high, generator=gen)
        param.data.copy_(w + noise.to(dtype=w.dtype))


@contextmanager
def temporary_uniform_weight_noise(
    model: nn.Module,
    *,
    alpha: float,
    generator: torch.Generator | None = None,
) -> Generator[None, None, None]:
    """Save eligible weights, add noise, ``yield``, **always** restore in ``finally``.

    Use around forward/backward; run ``optimizer.step()`` **after** the context
    exits so updates apply to clean weights.

    The ``finally`` block runs on success, user exceptions, and worker death — if
    restore were omitted, a noisy forward could leave weights corrupted for the
    next step.
    """
    state = _NoiseState.capture(model)
    try:
        add_uniform_noise_to_eligible_weights(model, alpha=alpha, generator=generator)
        yield
    finally:
        state.restore()


def verify_forbidden_params_untouched(
    model: nn.Module,
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> list[str]:
    """Return list of parameter names that changed but are not eligible (sanity)."""
    violations: list[str] = []
    for name, p in model.named_parameters():
        if name not in before:
            continue
        if torch.equal(before[name], after[name]):
            continue
        module_name, _, param_name = name.rpartition(".")
        if module_name:
            mod = model.get_submodule(module_name)
        else:
            mod = model
        if is_eligible_weight_param(mod, param_name):
            continue
        violations.append(name)
    return violations


def snapshot_parameter_data(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone all parameter tensors for diff checks."""
    return {k: v.data.detach().clone() for k, v in model.named_parameters()}
