# Pre-flight VRAM estimation

Before renting the GPU, ask whether the run fits:

```bash
$ torch-preflight estimate finetune.py --gpu a100-80gb

Model      llama-2-7b  (arch-snapshot)   6.74 B params
Config     amp · AdamW · batch 4 · seq 2048
Read from  batch_size=finetune.py:9, optimizer=finetune.py:7,
           precision=finetune.py:15, seq_len=finetune.py:13

  weights            25.10 GiB  ██
  gradients          25.10 GiB  ██
  optimizer state    50.21 GiB  ███
  activations       114.00 GiB  ███████████████
  CUDA context         600 MiB  █
  fragmentation      21.44 GiB  ██
  ──────────────────────────────────────────────
  projected peak    236.44 GiB   (212.79 GiB – 260.08 GiB)

Target     NVIDIA A100 80GB (78.0 GiB usable)   →   303% of capacity   ✗ OOM

What would make it fit:
  ✗  − 88.00 GiB  →  148.44 GiB   flash attention / SDPA
       mathematically equivalent, removes the O(seq²) attention term
  ✗  −119.28 GiB  →  117.16 GiB   gradient checkpointing
       same result, roughly 30% slower
  ✗  − 41.42 GiB  →  195.02 GiB   8-bit AdamW (bitsandbytes)
       quantised optimizer state, minimal quality impact
  ~  −165.13 GiB  →   71.30 GiB   flash attention / SDPA + gradient checkpointing
                                  + 8-bit AdamW + halve micro-batch (2x accumulation)
       lands inside the card but not inside the error margin — it may well run,
       but there is no headroom for a fragmented allocator
```

`✓` fits with the error margin, `~` fits the point estimate but not the margin, `✗` does
not fit. A single 80GB A100 is genuinely the wrong tool for a full 7B fine-tune at
sequence 2048 — and the point is to learn that before renting one.

The model, batch size, sequence length, precision, optimizer and sharding strategy are all
read out of the script — nothing is imported or executed. Override any of them:

```bash
torch-preflight estimate train.py --gpu rtx4090 --batch-size 1 --seq-len 512 --dtype pure-bf16
torch-preflight estimate --model llama-2-7b --gpu 8xa100-80gb --sharding zero-3
torch-preflight estimate --params 13B --gpu-memory 48GiB
torch-preflight gpus --instances        # every known GPU and cloud instance
```

Cloud instance names work directly: `--gpu p4de.24xlarge`, `--gpu ml.p5.48xlarge`,
`--gpu a2-ultragpu-8g`.

## Install extras

```bash
pip install torch-preflight              # linter + static VRAM estimation, no torch needed
pip install "torch-preflight[hub]"       # + look up unknown architectures on the HF hub
pip install "torch-preflight[vram]"      # + exact meta-device profiling
```

Extras add *dependencies only* — the wheel is byte-identical either way, and the base
install never imports torch.

## Custom architectures

Models outside the bundled snapshot can be measured exactly, with `pip install
"torch-preflight[vram]"`:

```bash
torch-preflight estimate --model mypkg.models:build_gpt \
    --model-args layers=24 --model-args hidden=1024 \
    --gpu a100-80gb --batch-size 8 --seq-len 1024 --dtype amp
```

The model is instantiated on PyTorch's `meta` device — **zero bytes allocated, no GPU
required** — so parameter counts are exact rather than estimated, and a forward pass under
`saved_tensors_hooks` captures precisely the tensors autograd retains for backward. That
is activation memory by definition, not by formula.

This path imports and executes your code, so it is opt-in and explicit. `torch-preflight check`
never reaches it.

## As a CI gate

Declare the hardware your team trains on and TG010 fails the build when a config will not
fit it:

```toml
[tool.torch-preflight]
target_gpu = "rtx4090"
```

TG010 is deliberately conservative. It stays silent unless `target_gpu` is set, it needs a
model it can identify, it **never fires on a low-confidence estimate**, and it never
touches the network.

## How accurate is it?

```
peak = weights + gradients + optimizer state + master weights
     + activations + CUDA context + fragmentation
```

Parameter counts are exact for models in the bundled snapshot, and the analytic formula for
everything else is within ~1% of published counts (enforced by `tests/calibration/`).

The activation coefficients are **measured**, not assumed: `tests/calibration/measure_activations.py`
captures the tensors autograd actually retains via `saved_tensors_hooks` and fits them
against sequence length. It runs on the meta device, so it allocates zero bytes and needs
no GPU. That measurement showed the published constants are a midpoint of two regimes —
models with dropout retain three tensors of `b·a·s²` in the attention path, models without
retain one — so torch-preflight charges Llama-class models the cheaper rate they actually pay.

The allocator constants are measured on a real GPU, and end-to-end projections are checked
against measured peaks from actual training steps:

| | measured | estimated | error |
|---|---|---|---|
| GPT-2, batch 4 × seq 128 | 2.97 GiB | 3.04 GiB | +2.5% |
| GPT-2, batch 8 × seq 256 | 5.81 GiB | 5.09 GiB | −12.4% |
| BERT-base, batch 4 × seq 128 | 2.44 GiB | 2.39 GiB | −2.1% |
| BERT-base, batch 8 × seq 256 | 3.38 GiB | 3.33 GiB | −1.7% |
| DistilBERT, batch 4 × seq 128 | 1.53 GiB | 1.48 GiB | −3.4% |
| DistilBERT, batch 8 × seq 256 | 1.95 GiB | 1.94 GiB | −0.4% |
| ResNet-50, batch 16 × 224px | 1.24 GiB | 1.31 GiB | +5.6% |
| ResNet-50, batch 32 × 224px | 1.99 GiB | 2.02 GiB | +1.4% |

Mean absolute error 3.7%, on a Tesla T4. Regenerate with
`tests/calibration/measure_cuda.py --models`.

The ResNet-50 rows matter beyond their own accuracy: those activations were measured on the
meta device, which allocates nothing, and they predict a real allocator to within 6%. That
is the assumption the whole no-GPU-required approach rests on, now tested rather than
asserted.

Every estimate carries an interval, and the verdict bands account for it — there is no
fabricated "95% failure risk" probability, because there is no data to calibrate one
against. If the model cannot be identified, torch-preflight reports `UNKNOWN` rather than
guessing a parameter count.

**Known gaps**, tracked in [design/TODO.md](../design/TODO.md):

- The LM-head cost per logit element is **not batch-invariant** in the measurements —
  ~19.7 bytes at batch 4 against ~14.7 at batch 8, consistently across four vocabularies. A
  single constant cannot express that, so GPT-2 at batch 8 × seq 256 remains ~12% under.
  Encoder models, which have no LM head, land within 4%.
- Entry-point profiling measures the forward pass only, so the transient where a
  checkpointed layer is recomputed during backward is still modelled analytically.
- Calibration covers one GPU (T4) and four model families. `CUDA_CONTEXT_BYTES` in
  particular is a single data point; larger cards plausibly differ, and
  `hardware.Gpu.context_mib` exists to hold per-card numbers as they arrive.
- Encoder-decoder models (T5, Whisper) have parameter counts but no activation model.

## Guarding a run at runtime

The estimator answers "will this fit?" before the job is submitted. `VRAMGuard` answers it
*inside* the process, against the model that actually exists:

```python
from torch_preflight import VRAMGuard

with VRAMGuard(model, optimizer=optimizer, batch_size=32, seq_len=2048):
    train()
```

```
VramRiskError: torch-preflight: this configuration is projected to need 701 MiB
(456 MiB-946 MiB) on limit 256MiB, which has 0.2 GiB usable. Verdict: CERTAIN_OOM.
  breakdown: weights 128 MiB, gradients 128 MiB, optimizer state 256 MiB, ...
  smallest change that fits: 8-bit AdamW (bitsandbytes) + pure bf16 weights
```

Parameters, gradients, optimizer state and the autocast cache come from the live model and
are exact; the optimizer kind and precision are read off the objects you pass.

Activations are **measured from your module**, not guessed. Given a `seq_len`, an
`image_size` or an explicit `example_input`, the guard runs one forward pass against
meta-device parameters with `saved_tensors_hooks` attached: that captures exactly what
autograd would retain while allocating nothing and leaving your model untouched — same
device, same dtype, same mode, in a few milliseconds. Without a shape, or if the module
cannot run on meta tensors (a `.item()` in `forward`, a custom autograd function), the term
is reported unknown and the interval widens. It is never assumed to be zero; for a
ResNet-50 at batch 32 the activations outweigh everything else combined, and a guard that
under-counts them stays quiet through exactly the OOM it was installed to catch.

It **raises only when the run cannot fit even at the optimistic end of the interval** —
anything less certain is a warning, because aborting a training job on a guess is worse
than the OOM it was trying to prevent. `strict=True` opts into raising on likely failures
too.

Needs `pip install "torch-preflight[vram]"`. On exit, `guard.measured_peak` and
`guard.accuracy` compare the projection against what the run actually used.

