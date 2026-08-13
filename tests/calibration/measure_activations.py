"""Measure the activation coefficients the cost model relies on.

Run this to regenerate the tracked fixture ``tests/calibration/measured_activations.json``:

    pip install torch numpy
    python tests/calibration/measure_activations.py

Method
------
``torch.autograd.graph.saved_tensors_hooks`` captures exactly the tensors autograd
retains for the backward pass — which *is* activation memory, by definition rather than
by approximation.

Everything runs on ``torch.device("meta")``, which allocates zero bytes, so the sweep can
use realistic hidden sizes and sequence lengths on a laptop with no GPU. Storages are
deduplicated by object identity (``storage._cdata``) because views alias one buffer;
``data_ptr()`` is 0 for every meta tensor and cannot be used. Spike 0001 verified that
meta and CPU produce byte-identical results with this key.

The cost model expresses per-layer activation bytes as

    s * b * h * (ACT_LINEAR_COEFF + ACT_ATTN_COEFF * a * s / h)

which is linear plus quadratic in the sequence length. Sweeping ``s`` and fitting
``alpha * s + beta * s^2`` recovers both coefficients directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd.graph import saved_tensors_hooks
except ImportError as exc:  # pragma: no cover - the whole script needs torch
    sys.exit(f"this script needs torch and numpy ({exc}): pip install torch numpy")


# ------------------------------------------------------------------ model under test


class Block(nn.Module):
    """A transformer block.

    ``realistic=True`` adds the biases and dropout that production models carry and the
    published accounting assumes; the minimal variant is the floor.
    """

    def __init__(self, h, a, i, realistic=False):
        super().__init__()
        self.a = a
        self.realistic = realistic
        bias = realistic
        self.ln1 = nn.LayerNorm(h)
        self.q = nn.Linear(h, h, bias=bias)
        self.k = nn.Linear(h, h, bias=bias)
        self.v = nn.Linear(h, h, bias=bias)
        self.o = nn.Linear(h, h, bias=bias)
        self.ln2 = nn.LayerNorm(h)
        self.fc1 = nn.Linear(h, i, bias=bias)
        self.fc2 = nn.Linear(i, h, bias=bias)
        self.drop = nn.Dropout(0.1) if realistic else nn.Identity()

    def attn(self, x, flash):
        b, s, h = x.shape
        d = h // self.a
        q = self.q(x).view(b, s, self.a, d).transpose(1, 2)
        k = self.k(x).view(b, s, self.a, d).transpose(1, 2)
        v = self.v(x).view(b, s, self.a, d).transpose(1, 2)
        if flash:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
            probs = self.drop(scores.softmax(dim=-1))
            out = probs @ v
        out = out.transpose(1, 2).reshape(b, s, h)
        return self.drop(self.o(out))

    def forward(self, x, flash=False):
        x = x + self.attn(self.ln1(x), flash)
        x = x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
        return x


class Stack(nn.Module):
    def __init__(self, layers, h, a, i, realistic=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(h, a, i, realistic) for _ in range(layers)]
        )

    def forward(self, x, flash=False, checkpointing=False):
        for block in self.blocks:
            if checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, flash, use_reentrant=False
                )
            else:
                x = block(x, flash)
        return x


# --------------------------------------------------------------------- measurement


def saved_bytes(model, x, *, flash=False, checkpointing=False) -> int:
    """Bytes autograd retains for backward, deduplicated by storage identity.

    Parameter storages are excluded. A ``Linear`` saves its weight so it can compute the
    input gradient, but that weight is the *same buffer* as the parameter — it is already
    counted in the weights term and is not activation memory. Including it inflates the
    linear coefficient by roughly 2x.
    """
    parameter_storages = {
        p.untyped_storage()._cdata for p in model.parameters()
    }
    storages = {}

    def pack(t):
        storage = t.untyped_storage()
        # ``_cdata`` is the storage object's identity. It works on meta tensors, where
        # ``data_ptr()`` is always 0. Verified equivalent to data_ptr on CPU.
        if storage._cdata in parameter_storages:
            return t
        storages[storage._cdata] = storage.nbytes()
        return t

    with saved_tensors_hooks(pack, lambda t: t):
        model(x, flash=flash, checkpointing=checkpointing).sum()

    return sum(storages.values())


def build(layers, h, a, i, batch, seq, realistic=False, device="meta"):
    with torch.device(device):
        model = Stack(layers, h, a, i, realistic)
    x = torch.randn(batch, seq, h, device=device, requires_grad=True)
    return model, x


def fit_quadratic(seqs, values):
    """Least-squares fit of ``c + alpha * s + beta * s^2``.

    The constant term matters: without it, any sequence-independent bytes get absorbed
    into ``alpha`` and silently inflate the linear coefficient. It is returned so the
    caller can assert it is negligible.
    """
    s = np.array(seqs, dtype=float)
    design = np.stack([np.ones_like(s), s, s ** 2], axis=1)
    solution, *_ = np.linalg.lstsq(design, np.array(values, dtype=float), rcond=None)
    return float(solution[0]), float(solution[1]), float(solution[2])


def sweep(layers, h, a, i, batch, seqs, realistic, flash=False):
    return [
        saved_bytes(*build(layers, h, a, i, batch, s, realistic), flash=flash)
        for s in seqs
    ]


def coefficients(layers, h, a, i, batch, seqs, realistic):
    """Recover ACT_LINEAR_COEFF and ACT_ATTN_COEFF from a sequence-length sweep."""
    measured = sweep(layers, h, a, i, batch, seqs, realistic)
    constant, alpha, beta = fit_quadratic(seqs, measured)

    # Activations here are fp32 (4 bytes); the published constants assume 2-byte, so
    # normalise by dtype to compare like with like.
    dtype_scale = 4 / 2
    linear = alpha / (layers * batch * h * dtype_scale)
    attention = beta / (layers * batch * a * dtype_scale)
    return linear, attention, constant, dict(zip(map(str, seqs), measured))


# --------------------------------------------------------------------------- runner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "measured_activations.json",
    )
    args = parser.parse_args()

    layers, h, a, i, batch = 4, 1024, 16, 4096, 2
    seqs = [128, 256, 512, 1024, 2048]

    print(f"torch {torch.__version__}  device=meta (zero allocation)")
    print(f"config: layers={layers} hidden={h} heads={a} intermediate={i} batch={batch}")

    # Cross-check that meta agrees with a real device before trusting the sweep.
    meta_bytes = saved_bytes(*build(2, 256, 8, 1024, 2, 128, device="meta"))
    cpu_bytes = saved_bytes(*build(2, 256, 8, 1024, 2, 128, device="cpu"))
    print(f"\nmeta/cpu agreement: {meta_bytes:,} vs {cpu_bytes:,} "
          f"-> {'MATCH' if meta_bytes == cpu_bytes else 'MISMATCH'}")

    results = {}
    for label, realistic in (("minimal", False), ("realistic", True)):
        linear, attention, constant, raw = coefficients(
            layers, h, a, i, batch, seqs, realistic
        )
        results[label] = {
            "act_linear_coeff": round(linear, 2),
            "act_attn_coeff": round(attention, 2),
            "fit_constant_bytes": round(constant),
            "measured_bytes_by_seq": raw,
        }
        print(f"\n{label:<10} ACT_LINEAR_COEFF = {linear:6.2f}"
              f"   ACT_ATTN_COEFF = {attention:5.2f}"
              f"   fit constant = {constant / 1e6:+.2f} MB")

    # Flash attention must remove the quadratic term entirely.
    flash_measured = sweep(layers, h, a, i, batch, seqs, True, flash=True)
    _, flash_alpha, flash_beta = fit_quadratic(seqs, flash_measured)
    flash_linear = flash_alpha / (layers * batch * h * 2)
    results["flash"] = {
        "act_linear_coeff": round(flash_linear, 2),
        "residual_quadratic": flash_beta,
        "measured_bytes_by_seq": dict(zip(map(str, seqs), flash_measured)),
    }
    print(f"\nflash/SDPA ACT_LINEAR_COEFF = {flash_linear:6.2f}"
          f"   residual quadratic = {flash_beta:.3g} (expect ~0)")

    # Checkpointing: forward-only, so this is the *stored* term. The recompute
    # transient is not visible here and is modelled separately.
    normal = saved_bytes(*build(layers, h, a, i, batch, 512, True))
    checkpointed = saved_bytes(
        *build(layers, h, a, i, batch, 512, True), checkpointing=True
    )
    stored_coeff = checkpointed / (layers * batch * 512 * h * 2)
    results["checkpointing"] = {
        "normal_bytes": normal,
        "checkpointed_bytes": checkpointed,
        "reduction": round(1 - checkpointed / normal, 4),
        "checkpoint_act_coeff": round(stored_coeff, 2),
        "note": "forward-only; excludes the one-layer recompute transient during backward",
    }
    print(f"\ncheckpointing: {normal:,} -> {checkpointed:,} "
          f"({(1 - checkpointed / normal) * 100:.1f}% less), "
          f"CHECKPOINT_ACT_COEFF = {stored_coeff:.2f}")

    payload = {
        "torch_version": torch.__version__,
        "method": "saved_tensors_hooks on torch.device('meta'), deduped by storage._cdata",
        "sweep": {
            "layers": layers, "hidden": h, "heads": a,
            "intermediate": i, "batch": batch, "seqs": seqs,
        },
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
