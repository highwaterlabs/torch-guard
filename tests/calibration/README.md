# Calibration fixtures

> **Where things live.** The measurement *scripts* are in `design/experimentations/`, which is
> gitignored local tooling. The *fixtures* they produce live here and are tracked, because
> the test suite reads them to pin the constants in `costmodel.py`. Regenerating a fixture
> means running the script in `design/`; the output lands here.

The cost model is built on empirical constants — fragmentation percentage, CUDA context
size, the Megatron activation coefficients. They are not derived truths, they are numbers
that are right in some regimes and wrong in others.

This directory is how we find out which. Without it the estimator is untested arithmetic
that happens to produce confident-looking output.

## What is measured today

| Fixture | Source of truth | Tolerance | Status |
|---|---|---|---|
| `param_counts.json` | Published parameter counts | 3% | ✅ enforced |
| Precision/optimizer accounting | Definitional arithmetic (4 + 4 + 8 bytes/param) | exact | ✅ enforced |
| `measured_activations.json` | `saved_tensors_hooks` on the meta device | ±0.5 on each coefficient | ✅ enforced |
| `measured_cnn_activations.json` | `saved_tensors_hooks` on the meta device | exact | ✅ enforced |
| LM-head retained bytes | `saved_tensors_hooks`, 5 shapes x 3 vocabularies | exact | ✅ enforced |
| Peak memory of a real run | `torch.cuda.max_memory_allocated()` | 25% | ✅ 6 runs on a T4 |

## Activation coefficients — measured

`measure_activations.py` captures the tensors autograd actually retains and fits
`alpha*s + beta*s^2` across a sequence-length sweep, recovering both coefficients
directly. It runs on `torch.device("meta")`, so it allocates **zero bytes** and needs no
GPU — a laptop is enough.

```bash
pip install torch
python tests/calibration/measure_activations.py
```

Results (torch 2.13). Dropout is the discriminator: it retains the mask and the dropout
output alongside the softmax output, tripling the quadratic term.

| | `ACT_LINEAR_COEFF` | `ACT_ATTN_COEFF` |
|---|---|---|
| no dropout (Llama, Mistral, Qwen) | 32.0 | 2.0 |
| dropout p>0 (BERT, GPT-2, RoBERTa) | 36.0 | 6.0 |

`CHECKPOINT_ACT_COEFF` measured at exactly 2.00, and flash attention shows no residual
quadratic term. Full write-up in [spike 0001](../../design/spikes/0001-meta-device-activation-capture.md).

**Two traps this harness guards against**, both of which produced badly wrong numbers on
the first run:

- Linear layers save their **weights** for the backward pass. Those are the same buffers as
  the parameters, already counted in the weights term — including them inflates the linear
  coefficient by ~2x. Parameter storages must be excluded.
- The fit needs a **constant term**. Without one, any sequence-independent bytes get
  absorbed into the linear coefficient silently. `test_measurement_fit_had_no_unexplained_constant`
  asserts the fitted constant stays at zero.

## Vision models

CNN activation memory has no closed form — it is the sum of every feature map, specific to
each architecture. So the snapshot stores a measured per-sample figure instead, produced by
`measure_cnn_activations.py` on the meta device (zero allocation, no GPU).

Two invariants the cost model depends on are checked at measurement time rather than
assumed: activations are linear in batch size, and scale with spatial area. Both hold for
all eleven models.

Worth knowing why this could not be derived: MobileNet-V2 has 3.5M parameters and *more*
activation memory than VGG-16's 138M. Any formula keyed on parameter count would be badly
wrong in both directions.

## The LM head: measured and fitted, kept separate

`cross_entropy` retains exactly **4.00 bytes per logit element** — one fp32 copy — measured
across five shapes and three vocabularies. The head projection itself retains ~0.02, since
its input is the hidden state already counted elsewhere.

The peak is larger than what is retained, because the backward pass holds the logits
gradient and the softmax workspace simultaneously, and forward-only capture cannot see
that. That part is a **fitted** constant supported by two data points, kept in its own
named constant so the weak evidence is visible.

Sweeping it against the measured peaks lowers mean error monotonically all the way to
18 bytes/element. That is the signature of absorbing a systematic under-estimate into
whichever parameter scales with `b*s*vocab`, not of finding a true value — so it was left
at the operation-count estimate rather than tuned to flatter six fixtures.

## How stable are these measurements?

An independent re-run on a fresh T4 reproduced all six peaks to within **0.51%**, and gave
an identical 135 MiB CUDA context. Fragmentation across the real models came out at 10.5%,
matching the shipped constant exactly.

That matters for interpreting a future disagreement: measurement noise here is well under
one percent, so a several-percent gap is a real modelling error rather than jitter.

## The remaining gap — and how to close it

Two constants still cannot be measured without NVIDIA hardware: `CUDA_CONTEXT_BYTES` and
`FRAGMENTATION_FRACTION`. Both are CUDA-allocator properties with no equivalent on MPS or
CPU. Until `measured_peaks.json` has entries they remain inherited from commonly reported
behaviour — which is part of why the estimator reports an interval and never a bare number.

`measure_cuda.py` closes both. **A free Colab or Kaggle T4 is enough** — no paid GPU
needed, and the whole run takes a few minutes.

### On Colab

Runtime → Change runtime type → **T4 GPU**, then in one cell:

```python
!pip install -q torch-preflight transformers
!wget -qO measure_cuda.py "https://raw.githubusercontent.com/highwaterlabs/torch-preflight/main/tests/calibration/measure_cuda.py?$(date +%s)"
!python measure_cuda.py --models
```

Two download traps, both of which have already cost a wasted run:

- **`-O` is required.** Plain `wget` refuses to overwrite an existing file; it writes
  `measure_cuda.py.1` and leaves the old one, so re-running in the same session silently
  executes the previous version.
- **The `?$(date +%s)` defeats the CDN.** `raw.githubusercontent.com` caches for a few
  minutes, so a download immediately after merging can still serve the old file.

If a run produces no `LM-head` or `VRAMGuard` section, it used a stale script.

Pasting the file's contents straight into a cell works too.

### What it measures

0. **Everything below in one run.** `--models` additionally validates the vision
   activations against a real peak, sweeps the LM-head cost, and checks `VRAMGuard`.

1. **CUDA context**, sampled three times — after `cuda.init()`, after the first matmul
   (which pulls in cuBLAS) and after the first convolution (cuDNN, the larger of the two).
   Computed as `(total - free)` from the driver minus what PyTorch has reserved, so it
   isolates memory that `torch.cuda.memory_allocated()` never sees.
2. **Fragmentation**, as `max_memory_reserved / max_memory_allocated` across five shapes.
3. **End-to-end peaks** for GPT-2, BERT, DistilBERT and ResNet-50, emitted as
   ready-to-paste JSON. ResNet-50 matters specifically: the CNN activation figures were
   measured on the meta device and have never been checked against a real allocator.
4. **An LM-head vocabulary sweep** — same tiny transformer body, vocabulary varied from
   8k to 128k at two batch sizes. Holding the body constant makes the `b*s*vocab`
   coefficient separable, which two data points could not do. The printed slope is the
   per-logit cost with backward transients included; compare it against
   `LM_HEAD_RETAINED_BYTES + LM_HEAD_BACKWARD_TRANSIENT_BYTES`.
5. **`VRAMGuard` accuracy** — projection versus what a real ResNet-50 step actually used,
   which is the runtime half of the same question.

Two details that would otherwise corrupt the numbers, both handled:

- A **warmup step is required**. Adam allocates its state lazily inside the first
  `.step()`, so measuring step 1 would miss the optimizer term entirely. The harness runs
  a warmup step, resets the peak counters, then measures a steady-state step.
- The comparable figure is `max_memory_reserved + cuda_context`, not `max_memory_allocated`.
  Reserved already includes fragmentation, and the context sits outside PyTorch's
  accounting — together they are what actually occupies the card.

The harness is exercised on CPU by `tests/test_calibration.py`, so a broken forward pass
shows up here rather than after you have pasted it into a notebook.

### Recording the result

Paste the printed block into `measured_peaks.json` under `"runs"`, and update
`CUDA_CONTEXT_BYTES` / `FRAGMENTATION_FRACTION` in `costmodel.py` if the measurement
disagrees. Entries look like this:

```json
{
  "model": "llama-2-7b",
  "gpu": "a100-80gb",
  "config": {
    "batch_size": 1, "seq_len": 2048, "precision": "amp",
    "optimizer": "adamw", "gradient_checkpointing": true, "flash_attention": true
  },
  "measured_peak_bytes": 42949672960,
  "torch_version": "2.4.0",
  "note": "single A100, no sharding"
}
```

`test_calibration.py` picks it up automatically and asserts the estimate lands within
tolerance. If it does not, that is the point: either the fixture is wrong or a constant in
`costmodel.py` is, and the discrepancy is now visible instead of silent.

## Changing a constant

Any change to a CALIBRATION constant in `costmodel.py` should move a fixture here. A
constant tuned until one number looked right, with nothing recording why, is how a cost
model quietly rots. Loosening a tolerance needs a reason in the commit message.
