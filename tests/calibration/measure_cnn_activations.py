"""Measure activation memory for the vision models in the bundled snapshot.

Run this to regenerate ``measured_cnn_activations.json``:

    pip install torch torchvision
    python tests/calibration/measure_cnn_activations.py

Why this exists
---------------
CNN activation memory has no tidy closed form the way a transformer's does — it is the sum
of every feature map, which depends on the specific architecture. So the cost model reads
a per-sample figure from the snapshot instead, and until now that figure was missing:
vision models reported activations as *unknown* and widened the interval rather than
guessing.

The same technique that measures transformers works here.
``torch.autograd.graph.saved_tensors_hooks`` on ``torch.device("meta")`` captures exactly
the tensors autograd retains, allocating nothing, so a laptop can measure a ResNet-152 at
any resolution it likes.

Two invariants are checked rather than assumed, because the cost model relies on both:

* activations scale **linearly with batch size**, so a per-sample number is meaningful;
* they scale with **spatial area**, so ``(size / reference) ** 2`` is the right way to
  rescale a measurement taken at 224px.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    import torchvision.models as tvm
    from torch.autograd.graph import saved_tensors_hooks
except ImportError as exc:  # pragma: no cover
    sys.exit(f"this script needs torch and torchvision ({exc}): pip install torch torchvision")

#: Snapshot key -> torchvision constructor. Only models the snapshot actually ships.
MODELS = {
    "resnet18": tvm.resnet18,
    "resnet34": tvm.resnet34,
    "resnet50": tvm.resnet50,
    "resnet101": tvm.resnet101,
    "resnet152": tvm.resnet152,
    "vgg16": tvm.vgg16,
    "vgg19": tvm.vgg19,
    "densenet121": tvm.densenet121,
    "mobilenet_v2": tvm.mobilenet_v2,
    "efficientnet_b0": tvm.efficientnet_b0,
    "convnext_tiny": tvm.convnext_tiny,
}

REFERENCE_SIZE = 224
#: The cost model expresses activations against a 2-byte reference and rescales by the
#: configured precision. Meta models are fp32, so halve the measurement.
DTYPE_NORMALISER = 2


def saved_bytes(model, batch: int, size: int) -> int:
    """Bytes autograd retains for backward, excluding parameters and buffers."""
    excluded = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        try:
            excluded.add(tensor.untyped_storage()._cdata)
        except Exception:
            continue

    storages = {}

    def pack(tensor):
        try:
            storage = tensor.untyped_storage()
        except Exception:
            return tensor
        # ``_cdata`` is storage identity: views alias one buffer, and every meta tensor
        # shares a null data_ptr, so that cannot be used as the key.
        if storage._cdata not in excluded:
            storages[storage._cdata] = storage.nbytes()
        return tensor

    model.train()
    x = torch.zeros(batch, 3, size, size, device="meta")
    with torch.enable_grad(), saved_tensors_hooks(pack, lambda t: t):
        model(x)
    return sum(storages.values())


def build(factory):
    with torch.device("meta"):
        return factory()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "measured_cnn_activations.json")
    args = parser.parse_args()

    print(f"torch {torch.__version__}  device=meta (zero allocation)\n")
    print(f"{'model':<18}{'params':>14}{'act/sample @224':>18}{'batch-linear':>14}"
          f"{'area-scaled':>13}")
    print("-" * 77)

    results = {}
    for name, factory in MODELS.items():
        model = build(factory)
        params = sum(p.numel() for p in model.parameters())

        one = saved_bytes(model, 1, REFERENCE_SIZE)
        four = saved_bytes(model, 4, REFERENCE_SIZE)
        linear = abs(four - 4 * one) / (4 * one) < 0.02

        # Does area scaling hold? 320px should cost (320/224)^2 as much.
        big = saved_bytes(model, 1, 320)
        expected = one * (320 / REFERENCE_SIZE) ** 2
        area = abs(big - expected) / expected < 0.10

        per_sample = int(one / DTYPE_NORMALISER)
        results[name] = {
            "params": params,
            "activation_bytes_per_sample": per_sample,
            "reference_image_size": REFERENCE_SIZE,
            "raw_fp32_bytes_batch1": one,
            "batch_linear": linear,
            "area_scaled": area,
        }
        print(f"{name:<18}{params:>14,}{one/1e6:>15.1f} MB"
              f"{'yes' if linear else 'NO':>14}{'yes' if area else 'NO':>13}")

    payload = {
        "torch_version": torch.__version__,
        "method": "saved_tensors_hooks on torch.device('meta'), parameters excluded, "
                  "deduped by storage._cdata; normalised to a 2-byte activation reference",
        "reference_image_size": REFERENCE_SIZE,
        "models": results,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")

    failures = [n for n, r in results.items() if not (r["batch_linear"] and r["area_scaled"])]
    if failures:
        print(f"\nWARNING: scaling assumptions do not hold for: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
