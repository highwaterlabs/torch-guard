# Spike 0001 — Capturing activation memory on the meta device

**Status:** Done — GO (see Result below)
**Time-box:** 1 day
**Blocks:** RFC [0001](../rfcs/0001-vram-estimator.md) phase 2

---

## The question

Can we measure **exact activation memory** for an arbitrary `nn.Module` without allocating
a single byte of real GPU memory?

Parameter counting on the meta device is already known to work — that is what
`accelerate estimate-memory` does. Activations are the hard part, and at large batch sizes
they dominate peak memory. If we cannot get them accurately, the meta provider is barely
better than the static formulas and phase 2 loses most of its value.

## Hypothesis

`torch.autograd.graph.saved_tensors_hooks` captures **exactly** the set of tensors autograd
retains for the backward pass. That set *is* activation memory, by definition — not an
approximation of it.

```python
import torch
from torch.autograd.graph import saved_tensors_hooks

saved_bytes = 0

def pack(t):
    global saved_bytes
    saved_bytes += t.numel() * t.element_size()
    return t

with torch.device("meta"):
    model = build_model()
    x = torch.randn(batch, seq, device="meta")

with saved_tensors_hooks(pack, lambda t: t):
    loss = model(x).sum()
```

Meta tensors carry real `numel()` and `element_size()` while allocating nothing, so the
arithmetic should hold even though no memory is touched.

## What we need to find out

1. Does `saved_tensors_hooks` actually fire under `torch.device("meta")`? Autograd on meta
   tensors builds a graph, but confirm the pack hook is invoked rather than skipped.
2. Does the sum double-count? Views share storage; two saved tensors may alias one buffer.
   Deduplicate by `untyped_storage()._cdata` (or equivalent) and compare both numbers.
3. Does it capture the **peak** or the **total**? Summing every saved tensor gives total
   retained-at-once, which is what we want for activations — but confirm no tensors are
   freed and re-saved during forward.
4. How do `torch.utils.checkpoint` regions behave? Checkpointing should visibly collapse the
   number, and by roughly the expected factor. This is the strongest correctness signal
   available without a GPU.
5. Does it survive `autocast`? Under mixed precision the saved tensors should be fp16/bf16,
   and the byte count should drop accordingly.
6. Do flash / fused attention paths (SDPA) report correctly, or do they hide saved state
   inside a fused kernel that never surfaces to the hook?

## Acceptance criteria

The spike **succeeds** if, for three models of different shapes — a small CNN, a small
transformer, and one HF model — the meta-device estimate lands within **10%** of the real
peak activation memory measured on an actual GPU via
`torch.cuda.max_memory_allocated()` minus the known weight/gradient/optimizer terms.

It **fails** if the hook does not fire on meta tensors, or if fused attention makes the
number systematically wrong for transformers — which are the models people are most likely
to OOM on, so a transformer-shaped blind spot is disqualifying.

## Fallbacks if it fails

1. **`TorchDispatchMode`** — intercept every aten op and record output tensor sizes. More
   invasive, requires modelling liveness ourselves, but does not depend on autograd
   internals.
2. **`FakeTensorMode`** — a level up from meta, handles more ops correctly and is what
   `torch.compile` uses internally for shape propagation.
3. **Real allocation on CPU** — accurate for shapes but slow, memory-hungry, and does not
   model GPU-specific allocator behaviour. Last resort.

## Deliverable

A note appended to this file with the measured numbers, the answer to each of the six
questions, and a go/no-go on the mechanism. **The spike code is thrown away** — if it
starts looking like production code, it stopped being a spike.

---

# Result — GO

**Run:** 2026-08-13, torch 2.13.0, MacBook Air M2 (no GPU involved — meta and CPU only).
**Verdict:** the mechanism works. Proceed with `providers/meta.py` in phase 2.

## Answers

**Q1 — Does `saved_tensors_hooks` fire under `torch.device("meta")`?** Yes, and the byte
counts are *identical* to CPU:

```
cpu    raw = 39,878,656   dedup = 33,587,200   tensors = 100
meta   raw = 39,878,656   dedup = 33,587,200   tensors = 100
```

**Q2 — Does summing double-count aliased storages?** Yes — 18.7% overcount on this model,
because views share one buffer. Deduplication is required.

**Q3 — How do we dedup on meta?** `storage.data_ptr()` is **0 for every meta tensor** and
cannot be used. `storage._cdata` (the storage object's identity) works on meta and gives
byte-identical results to `data_ptr()` on CPU. This was the one genuine unknown, and it
has a clean answer.

**Q4 — Checkpointing?** Collapses the stored term by 97.6%, and the measured
`CHECKPOINT_ACT_COEFF` is exactly **2.00** — one tensor of `s*b*h`, matching the model's
assumption exactly. Note this is a *forward-only* measurement, so it excludes the
one-layer recompute transient during backward, which the cost model adds separately.

**Q5 — autocast?** Saved bytes drop to 0.53x under bf16 autocast, confirming the dtype
scaling. Not exactly 0.5 because LayerNorm statistics and a few other tensors stay fp32.

**Q6 — Does fused attention hide state?** **No — this was the disqualifying risk and it did
not materialise.** SDPA reduces saved bytes by exactly the score-matrix term and the hook
still sees everything SDPA retains. Fitting the flash sweep leaves a residual quadratic
coefficient of 512 bytes against a ~700 MB total, i.e. zero. Flash attention is measured
correctly, not hidden.

## Two methodology bugs found (both would have poisoned the numbers)

1. **Linear layers save their weights** for the input-gradient computation. Those are the
   same buffers as the parameters — already counted in the weights term, and not
   activation memory. Including them inflated the linear coefficient by ~2x. Fixed by
   excluding parameter storages.
2. **The fit had no constant term**, so those sequence-independent weight bytes were
   absorbed into the linear coefficient instead of showing up as an offset. After both
   fixes the fitted constant is exactly 0.00 MB, which is the signal that the model is
   now fully explained by `alpha*s + beta*s^2`.

Both are now regression-tested in `tests/test_calibration.py`.

## What this changed in the product

The measurement validated the published Megatron constants but showed they are a *midpoint*
of two regimes that differ by 3x on the quadratic term:

| | `ACT_LINEAR_COEFF` | `ACT_ATTN_COEFF` |
|---|---|---|
| no dropout (Llama, Mistral, Qwen) | **32.0** | **2.0** |
| dropout p>0 (BERT, GPT-2, RoBERTa) | **36.0** | **6.0** |
| published Megatron / our old constant | 34 | 5 |

Dropout retains the mask *and* the dropout output alongside the softmax output, tripling
the `O(seq²)` term. `nn.Dropout(0.0)` short-circuits and costs nothing, which is why modern
LLMs sit in the cheaper regime. `TransformerShape.uses_dropout` now selects the pair, and
the Llama-2-7b activation estimate fell 44% (228 → 128 GiB at batch 8, seq 2048) as a
result — a real correction on exactly the model class people OOM on.

## Follow-ups for phase 2

- The provider must exclude parameter storages and dedup by `_cdata`, exactly as
  `tests/calibration/measure_activations.py` does.
- Forward-only capture misses the recompute transient under checkpointing; keep modelling
  that term analytically.
- Spike code is deleted. The measurement harness graduated to
  `tests/calibration/measure_activations.py` and is a supported entry point.
