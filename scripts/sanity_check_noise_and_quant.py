"""Verify uniform weight noise and PTQ only touch eligible weights; quant w8/w6/w4 works.

Uses a tiny CPU model (no CompressAI). Optionally verifies ``results/`` is writable.

Run manually:

  python scripts/sanity_check_noise_and_quant.py
"""

from __future__ import annotations

import sys

from _bootstrap import ensure_package_on_path

ensure_package_on_path()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from weight_noise_ptq.common.noise import (  # noqa: E402
    add_uniform_noise_to_eligible_weights,
    is_eligible_weight_param,
    snapshot_parameter_data,
    temporary_uniform_weight_noise,
    verify_forbidden_params_untouched,
)
from weight_noise_ptq.common.paths import repo_root  # noqa: E402
from weight_noise_ptq.common.quantization import quantize_eligible_weights_in_model  # noqa: E402


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8 * 8 * 8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(torch.relu(self.conv(x)))
        x = x.flatten(1)
        return self.fc(x)


def _check_results_dir_writable() -> None:
    root = repo_root() / "results"
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".sanity_write_probe"
    probe.write_text("ok", encoding="utf-8")
    if probe.exists():
        probe.unlink()


def main() -> None:
    torch.manual_seed(0)
    m = _TinyNet()
    x = torch.randn(2, 3, 8, 8)
    before = snapshot_parameter_data(m)

    with temporary_uniform_weight_noise(m, alpha=0.02):
        y = m(x)
        assert y.shape == (2, 4)
    restored = snapshot_parameter_data(m)
    for k in before:
        assert torch.equal(before[k], restored[k]), k

    add_uniform_noise_to_eligible_weights(m, alpha=0.02)
    noisy = snapshot_parameter_data(m)
    bad = verify_forbidden_params_untouched(m, before, noisy)
    assert not bad, bad

    for name, p in m.named_parameters():
        mod_name, _, param = name.rpartition(".")
        mod = m if not mod_name else m.get_submodule(mod_name)
        if param == "weight" and isinstance(mod, (nn.Conv2d, nn.Linear)):
            assert is_eligible_weight_param(mod, "weight")

    for bw in ("w8", "w6", "w4"):
        qm = quantize_eligible_weights_in_model(m, bw)
        with torch.no_grad():
            yq = qm(x)
        assert yq.shape == (2, 4)

    _check_results_dir_writable()
    print("sanity_check_noise_and_quant: passed.", file=sys.stderr)


if __name__ == "__main__":
    main()
