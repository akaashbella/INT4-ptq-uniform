"""Verify all locked models instantiate and run a single forward pass (CPU).

Run manually on a login node or workstation after ``pip install -e .``:

  python scripts/sanity_check_models.py [--device cpu|cuda]

Does **not** load Tiny ImageNet or checkpoints.
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import ensure_package_on_path

ensure_package_on_path()

import torch  # noqa: E402

from weight_noise_ptq.classification.registry import build_classification_model  # noqa: E402
from weight_noise_ptq.compression.registry import build_compression_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    return p.parse_args()


def _check_classification(device: torch.device) -> None:
    for name in ("resnet50", "mobilenetv3_large", "convnext_tiny"):
        m = build_classification_model(name, num_classes=200, pretrained=False).to(device)
        m.eval()
        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 200), name
        print(f"OK classification {name} -> {tuple(y.shape)}")


def _check_compression(device: torch.device) -> None:
    for name in ("factorized_prior", "scale_hyperprior", "cheng2020_attention"):
        m = build_compression_model(name, quality=4, pretrained=False).to(device)
        m.eval()
        x = torch.rand(2, 3, 64, 64, device=device)
        with torch.no_grad():
            out = m(x)
        assert isinstance(out, dict) and "x_hat" in out, name
        print(f"OK compression {name} -> x_hat {tuple(out['x_hat'].shape)}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available.", file=sys.stderr)
        sys.exit(2)
    _check_classification(device)
    _check_compression(device)
    print("sanity_check_models: all passed.")


if __name__ == "__main__":
    main()
