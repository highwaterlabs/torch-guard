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
| Peak memory of a real run | `torch.cuda.max_memory_allocated()` | 25% | ⚠️ **not yet populated** |

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
!wget -q https://raw.githubusercontent.com/<repo>/main/tests/calibration/measure_cuda.py
!python measure_cuda.py --models
```

Pasting the file's contents straight into a cell works too.

### What it measures

1. **CUDA context**, sampled three times — after `cuda.init()`, after the first matmul
   (which pulls in cuBLAS) and after the first convolution (cuDNN, the larger of the two).
   Computed as `(total - free)` from the driver minus what PyTorch has reserved, so it
   isolates memory that `torch.cuda.memory_allocated()` never sees.
2. **Fragmentation**, as `max_memory_reserved / max_memory_allocated` across five shapes.
3. **End-to-end peaks** for GPT-2, BERT and DistilBERT, emitted as ready-to-paste JSON.

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
