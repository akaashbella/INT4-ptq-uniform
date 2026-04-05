"""Symmetric per-tensor weight-only quantization (evaluation copies; original untouched)."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

import torch
from torch import nn

from weight_noise_ptq.common.noise import is_eligible_weight_param, iter_eligible_weight_parameters

BitwidthName = Literal["fp32", "w8", "w6", "w4"]


def int_bits_for_label(label: BitwidthName | str) -> int | None:
    """Map ``w8``/``w6``/``w4`` to integer bitwidth; ``fp32`` -> ``None``."""
    s = str(label)
    if s == "fp32":
        return None
    if s == "w8":
        return 8
    if s == "w6":
        return 6
    if s == "w4":
        return 4
    raise ValueError(f"Unknown bitwidth label: {label}")


def _qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def quantize_tensor_symmetric_per_tensor(
    w: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Apply locked quantization formula elementwise on a single weight tensor.

    ``s = max(abs(W)) / (2^(b-1)-1)``; if ``max(abs(W))==0``, return zeros.
    ``W_q = clamp(round(W/s), -qmax, qmax) * s``.

    ``torch.round`` uses IEEE “round half to even” on ties; this matches the
    locked symmetric uniform PTQ definition in code (not stochastic rounding).
    """
    if bits < 2:
        raise ValueError("bits must be >= 2")
    w_f = w.detach().float()
    max_abs = w_f.abs().max()
    qmax = _qmax(bits)
    if max_abs.item() == 0.0:
        return torch.zeros_like(w)
    scale = max_abs / float(qmax)
    q = torch.round(w_f / scale).clamp(-float(qmax), float(qmax))
    out = (q * scale).to(dtype=w.dtype)
    return out


def quantize_eligible_weights_in_model(
    model: nn.Module,
    bitwidth: BitwidthName | str,
) -> nn.Module:
    """Deep-copy ``model`` and replace eligible weights with quantized tensors.

    Non-eligible parameters and buffers are copied unchanged. Does not mutate
    the input module.
    """
    label: BitwidthName = bitwidth  # type: ignore[assignment]
    bits = int_bits_for_label(label)
    m_copy = deepcopy(model)
    if bits is None:
        return m_copy
    with torch.no_grad():
        for _, param, _ in iter_eligible_weight_parameters(m_copy):
            q = quantize_tensor_symmetric_per_tensor(param.data, bits)
            param.data.copy_(q)
    return m_copy


def quantize_state_dict_eligible_only(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
    bitwidth: BitwidthName | str,
) -> dict[str, torch.Tensor]:
    """New state dict with quantized eligible weights; other keys copied."""
    bits = int_bits_for_label(bitwidth)
    out = {k: v.clone() for k, v in state_dict.items()}
    if bits is None:
        return out
    for name, module in model.named_modules():
        if not is_eligible_weight_param(module, "weight"):
            continue
        key = f"{name}.weight" if name else "weight"
        if key not in out:
            continue
        out[key] = quantize_tensor_symmetric_per_tensor(out[key], bits)
    return out
