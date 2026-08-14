"""Measure activation memory for encoder-decoder models (T5, Whisper).

Run this to regenerate ``measured_encoder_decoder.json``:

    pip install torch transformers
    python tests/calibration/measure_encoder_decoder.py

Why these need their own coefficients
-------------------------------------
A decoder-only stack has one sequence length. An encoder-decoder has two, and a decoder
layer carries a *third* attention block — cross-attention, whose score matrix is
``b·a·s_dec·s_enc`` and whose K/V projections run at the **encoder** length even though the
layer is a decoder layer. None of that is expressible in the decoder-only formula, which is
why these models previously reported activations as unknown.

The model fitted here is:

    encoder = L_enc · (enc_linear·h·s_enc + attn·a·s_enc²)
    decoder = L_dec · (dec_linear·h·s_dec + attn·a·s_dec²
                       + cross_attn·a·s_dec·s_enc + cross_kv·h·s_enc)

all times the batch size, in bytes at the 2-byte activation reference.

Collinearity, three times over
------------------------------
Every naive version of this fit is degenerate, and each one reports a *better* residual than
the correct version. That is worth stating plainly: on this problem, residual quality is
evidence of nothing.

**1. Encoder against cross-KV.** Fitting encoder and decoder together on T5 gives a 0.00%
residual and is wrong. T5 has ``num_layers == num_decoder_layers``, so ``L_enc·h·s_enc`` and
``L_dec·h·s_enc`` are the same column up to a constant; least squares splits the coefficient
evenly and calls it exact, because only the *sum* is identified. It returned enc_linear
26.84 where the truth is 48.34. Fixed by measuring the encoder **alone** first, where no
decoder term exists.

**2. Linear against quadratic.** Whisper requires exactly 3000 mel frames, so ``s_enc`` is
always 1500 and cannot be swept. Sweeping model sizes does not rescue it: every Whisper size
uses ``head_dim = 64``, so ``h·s_enc`` and ``a·s_enc²`` stay exactly proportional. This is
unidentifiable in principle, not merely ill-conditioned.

**3. Decoder-linear against cross-attention.** The same ``head_dim = 64`` with a fixed
``s_enc`` makes ``h·s_dec`` and ``a·s_dec·s_enc`` proportional too. Unconstrained, this
returned ``dec_linear = 0.16`` — a decoder layer retaining essentially nothing, which is not
a physical quantity.

Both quadratic terms are therefore **constrained** to the attention coefficient already
measured by ``measure_activations.py`` (6.0 with dropout, 2.0 without), leaving only linear
terms free. This is a constraint on a separately measured quantity rather than a fudge, and
T5 checks it: where the split *is* identifiable, cross-attention fits free at 6.03 against a
constrained 6.0. Cross-attention retains a ``b·a·s_dec·s_enc`` softmax output for exactly the
same reason self-attention retains ``b·a·s²``, so equating them is the physical claim, and
T5 is the evidence for it.

Sanity checks on the result, none of which the degenerate fits pass: ``dec_linear >
enc_linear`` in both families, because a decoder layer carries a third attention block; and
``cross_kv`` lands at 4.05 and 4.19 in two independently fitted families.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch.autograd.graph import saved_tensors_hooks
    from transformers import AutoConfig, AutoModel
except ImportError as exc:  # pragma: no cover
    sys.exit(f"this script needs torch, numpy and transformers ({exc})")

#: The cost model expresses activations against a 2-byte reference; meta models are fp32.
DTYPE_NORMALISER = 2

#: Measured by measure_activations.py. Dropout triples the retained attention tensors.
ATTN_COEFF_DROPOUT = 6.0
ATTN_COEFF_NO_DROPOUT = 2.0

#: Whisper's encoder is fixed: 3000 mel frames, halved by the stride-2 conv frontend.
WHISPER_ENCODER_SEQ = 1500
WHISPER_MEL_FRAMES = 3000

FAMILIES = {
    "t5": ["t5-small", "t5-base", "t5-large"],
    "whisper": ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"],
}


def build(name):
    config = AutoConfig.from_pretrained(name)
    with torch.device("meta"):
        model = AutoModel.from_config(config)
    model.train()
    excluded = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        try:
            excluded.add(tensor.untyped_storage()._cdata)
        except Exception:
            continue
    return config, model, excluded


def retained_bytes(call, excluded) -> int:
    """Bytes autograd retains for backward, excluding parameters and buffers."""
    storages = {}

    def pack(tensor):
        try:
            storage = tensor.untyped_storage()
        except Exception:
            return tensor
        if storage._cdata not in excluded:
            storages[storage._cdata] = storage.nbytes()
        return tensor

    with torch.enable_grad(), saved_tensors_hooks(pack, lambda t: t):
        call()
    return sum(storages.values())


def dims(config, family):
    if family == "whisper":
        return (config.encoder_layers, config.decoder_layers, config.d_model,
                config.encoder_attention_heads)
    return (config.num_layers, config.num_decoder_layers, config.d_model,
            config.num_heads)


def uses_dropout(config) -> bool:
    for field in ("dropout_rate", "dropout"):
        value = getattr(config, field, None)
        if value is not None:
            return float(value) > 0
    return False


def measure_family(family, names, verbose=True):
    """Fit the six coefficients for one architecture family."""
    enc_rows, enc_y, dec_rows, dec_y = [], [], [], []
    per_model = {}

    for name in names:
        config, model, excluded = build(name)
        layers_enc, layers_dec, hidden, heads = dims(config, family)
        attn_coeff = ATTN_COEFF_DROPOUT if uses_dropout(config) else ATTN_COEFF_NO_DROPOUT

        if family == "whisper":
            features = torch.zeros(1, config.num_mel_bins, WHISPER_MEL_FRAMES,
                                   device="meta")
            encoder_lengths = [WHISPER_ENCODER_SEQ]

            def encoder_call(_f=features, _m=model):
                return _m.encoder(input_features=_f)

            def full_call(seq_enc, seq_dec, _f=features, _m=model):
                ids = torch.zeros(1, seq_dec, dtype=torch.long, device="meta")
                return _m(input_features=_f, decoder_input_ids=ids)

            decoder_lengths = [32, 64, 96, 128]
        else:
            encoder_lengths = [64, 128, 192, 256]

            def encoder_call(seq_enc=None, _m=model):
                ids = torch.zeros(1, seq_enc, dtype=torch.long, device="meta")
                return _m.encoder(input_ids=ids)

            def full_call(seq_enc, seq_dec, _m=model):
                return _m(
                    input_ids=torch.zeros(1, seq_enc, dtype=torch.long, device="meta"),
                    decoder_input_ids=torch.zeros(1, seq_dec, dtype=torch.long,
                                                  device="meta"),
                )

            decoder_lengths = [64, 128, 192, 256]

        scale = DTYPE_NORMALISER / 4
        for seq_enc in encoder_lengths:
            if family == "whisper":
                encoder_only = retained_bytes(encoder_call, excluded)
            else:
                encoder_only = retained_bytes(lambda s=seq_enc: encoder_call(s), excluded)
            # Encoder measured alone: no decoder term exists, so its coefficients are
            # identified rather than split with the collinear cross-KV column.
            quadratic = attn_coeff * layers_enc * heads * seq_enc * seq_enc
            enc_rows.append([layers_enc * hidden * seq_enc])
            enc_y.append(encoder_only * scale - quadratic)

            for seq_dec in decoder_lengths:
                full = retained_bytes(
                    lambda e=seq_enc, d=seq_dec: full_call(e, d), excluded
                )
                decoder_only = (full - encoder_only) * scale
                dec_rows.append([
                    layers_dec * hidden * seq_dec,                    # dec_linear
                    layers_dec * hidden * seq_enc,                    # cross_kv
                ])
                # Both score matrices are constrained to the measured attention
                # coefficient. Cross-attention retains a softmax output of
                # b·a·s_dec·s_enc exactly as self-attention retains b·a·s², so this is
                # the same quantity, and T5 — where the split *is* identifiable —
                # measures it free at 6.03 against a constrained 6.0. Without the
                # constraint Whisper is unidentifiable: its encoder length is fixed and
                # every size uses head_dim 64, so the decoder-linear and cross-attention
                # columns stay exactly proportional and least squares splits them
                # arbitrarily (it returned dec_linear 0.16, which is not a real quantity).
                self_attention = attn_coeff * layers_dec * heads * seq_dec * seq_dec
                cross_attention = attn_coeff * layers_dec * heads * seq_dec * seq_enc
                dec_y.append(decoder_only - self_attention - cross_attention)

        per_model[name] = {
            "layers_enc": layers_enc, "layers_dec": layers_dec,
            "hidden": hidden, "heads": heads, "uses_dropout": uses_dropout(config),
        }
        if verbose:
            print(f"  {name:<24} L={layers_enc}/{layers_dec} h={hidden} a={heads}")

    enc_coeff, *_ = np.linalg.lstsq(np.array(enc_rows, float),
                                    np.array(enc_y, float), rcond=None)
    dec_coeff, *_ = np.linalg.lstsq(np.array(dec_rows, float),
                                    np.array(dec_y, float), rcond=None)

    enc_pred = np.array(enc_rows, float) @ enc_coeff
    dec_pred = np.array(dec_rows, float) @ dec_coeff
    return {
        "enc_linear": float(enc_coeff[0]),
        "dec_linear": float(dec_coeff[0]),
        "cross_kv": float(dec_coeff[1]),
        "attn": ATTN_COEFF_DROPOUT if any(
            m["uses_dropout"] for m in per_model.values()
        ) else ATTN_COEFF_NO_DROPOUT,
        "encoder_seq_len": WHISPER_ENCODER_SEQ if family == "whisper" else None,
        "models": per_model,
        "encoder_residual_max_pct": float(
            np.abs((enc_pred - np.array(enc_y)) / np.array(enc_y) * 100).max()
        ),
        "decoder_residual_max_pct": float(
            np.abs((dec_pred - np.array(dec_y)) / np.array(dec_y) * 100).max()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "measured_encoder_decoder.json")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    print(f"torch {torch.__version__}  device=meta (zero allocation)\n")
    results = {}
    for family, names in FAMILIES.items():
        print(f"{family}:")
        results[family] = measure_family(family, names)
        r = results[family]
        print(f"    enc_linear {r['enc_linear']:6.2f}   dec_linear {r['dec_linear']:6.2f}"
              f"   cross_kv {r['cross_kv']:5.2f}"
              f"   attn/cross_attn {r['attn']:.1f} (constrained)")
        print(f"    residuals: encoder {r['encoder_residual_max_pct']:.2f}%  "
              f"decoder {r['decoder_residual_max_pct']:.2f}%\n")

    payload = {
        "torch_version": torch.__version__,
        "method": "saved_tensors_hooks on torch.device('meta'); encoder measured alone to "
                  "break collinearity with the cross-KV term; the quadratic attention "
                  "coefficient is constrained to the separately measured 6.0/2.0 because "
                  "Whisper's fixed encoder length and constant head_dim make the "
                  "linear/quadratic split unidentifiable",
        "reference_dtype_bytes": DTYPE_NORMALISER,
        "families": results,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
