# Changelog

All notable changes to torch-preflight are recorded here. This project follows
[semantic versioning](https://semver.org/).

## [0.3.0] — 2026-08-14

Seven new rules, and a VRAM estimator that now covers serving as well as training.

### Added

**Rules — 6 to 13.**

- **TG006** binary cross-entropy paired with the wrong activation: `sigmoid` into
  `BCEWithLogitsLoss` (applied twice), raw logits into `BCELoss` (`nan` on the first
  negative value), and the numerically fragile-but-correct `sigmoid` + `BCELoss`, which
  warns rather than errors. Autofixable when the sigmoid is inline.
- **TG007** a GPU sync (`.item()`, `.cpu()`, `.numpy()`) inside a loop nested in the
  training step, or `torch.cuda.synchronize()` every step. One sync per step is *not*
  flagged — that is what TG001 tells you to write, and there is a test asserting the two
  rules never contradict each other.
- **TG008** a training run whose randomness is unseeded, or seeded for only some of torch /
  NumPy / `random`. Names which generator is missing; partial seeding is the usual shape.
- **TG011** `model.eval()` in an epoch loop with no matching `train()`, so only the first
  epoch trains with dropout on and batch-norm updating.
- **TG012** a `DataLoader` under DDP with no `DistributedSampler`: every rank iterates the
  whole dataset, so N GPUs train as one. Errors on training loaders, warns on evaluation.
- **TG013** a host-to-device transfer repeated every iteration — loop-invariant data, a
  `torch.*` host factory, or the model itself. Batch transfers are not flagged.
- **TG014** gradient accumulation without dividing the loss, which scales the summed
  gradient by the accumulation count — arithmetically the same as an N× learning rate.
  Autofix rewrites `loss.backward()` as `(loss / N).backward()`.

**Estimator.**

- **KV cache and generation sizing.** `--generate` and `--max-context` model autoregressive
  decoding: the cache is `2 · layers · kv_heads · head_dim · context · batch · dtype`, and
  it is where grouped-query attention pays off. Detected automatically from `.generate(...)`
  or `use_cache=True`.
- **Encoder-decoder models.** T5 and Whisper estimate activations instead of reporting
  unknown, including the cross-attention term a decoder-only formula cannot express.
  Validated to 2.5% worst case over 12 shapes, with two model sizes held out of the fit.
- **DeepSpeed config parsing.** The ZeRO stage is read from the dict or JSON your script
  points at rather than assumed to be stage 2, and `offload_optimizer` removes optimizer
  state and the fp32 master copy from the device.
- `tests/calibration/verify_snapshot.py` checks the bundled architecture snapshot against
  the live Hugging Face configs.

### Fixed

- **`--inference` charged an attention matrix that decoding never builds.** It ran the
  training activation formula, so a GPT-2 generation estimate at batch 32 and 4096 context
  read 105 GiB. Generation now costs a single decode step.
- **`Provenance.criteria` leaked across scopes.** Two functions each binding `criterion`
  collided and whichever was parsed last decided the loss class for both, so a correct
  `BCELoss` call could be reported as an error against `BCEWithLogitsLoss`. **This affected
  TG005.**
- The release workflow's `workflow_dispatch` rehearsal skipped the artifact download, so an
  incompatible upload/download pair would only have surfaced during a real release.
- Inference estimates no longer name an optimizer in the config summary.

### Changed

- Calibration extended to ResNet-50, confirming the meta-device activation measurements
  against a real allocator (+5.6% and +1.4%). Mean absolute error stays 3.7% across 8 runs.
- Grouped-query attention is deliberately **not** modelled as reducing training activations.
  Measured: `transformers.repeat_kv` materialises full-size K/V, so retained bytes are
  identical across `kv_heads` of 16, 8, 4 and 2. It does shrink the KV cache, which is
  modelled.
- GitHub Actions pins moved off the deprecated Node 20 runtime.

### Known gaps

Stated rather than papered over, and tracked as issues:

- `CUDA_CONTEXT_BYTES` is still a single Tesla T4 measurement ([#21](https://github.com/highwaterlabs/torch-preflight/issues/21)).
- The LM-head cost per logit is not batch-invariant, so GPT-2 at batch 8 stays ~12% under
  ([#22](https://github.com/highwaterlabs/torch-preflight/issues/22)).
- `offload_param` is detected but not subtracted, so those runs are over-estimated
  ([#24](https://github.com/highwaterlabs/torch-preflight/issues/24)).
- Paged-attention runtimes (vLLM, TensorRT-LLM) manage the KV cache in blocks; the estimate
  is the right order of magnitude, not their occupancy.

## [0.2.0] — 2026-08-13

### Fixed

- **`VRAMGuard` no longer treats activation memory as zero.** It profiled parameters
  exactly and had no way to see activations, so the term silently read zero — against a
  real ResNet-50 at batch 32 it projected 0.61 GiB where the card peaked at 1.86 GiB
  (−67.5%). Under-estimating is the direction that keeps a guard quiet through the OOM it
  exists to prevent. The guard now measures activations from the live module by running it
  against meta-device parameters, which allocates nothing and leaves the model untouched;
  the same case is now +8.8%. Pass `example_input` for models whose input shape cannot be
  derived from `seq_len` or `image_size`, or `measure_activations=False` to skip it. If the
  module cannot run on meta tensors the term is reported unknown, never zero.

### Changed

- LM-head backward transient raised from 6 to 10 bytes per logit element, replacing a
  two-point fit with a measured vocabulary sweep (8k–128k vocabularies at two batch sizes).
  Mean absolute error against measured peaks improves from 4.4% to 3.7%.
- Calibration fixtures extended with ResNet-50 peaks, confirming the meta-measured CNN
  activations against a real allocator (+5.6% and +1.4%).

## [0.1.0] — 2026-08-13

First release. [On PyPI](https://pypi.org/project/torch-preflight/0.1.0/).

### Linter

- **TG001** tensors stored with their autograd graph attached (`losses.append(loss)`,
  `total += loss`, `cache[k] = out`)
- **TG002** evaluation or inference running without `torch.no_grad()`
- **TG003** `.backward()` in a loop with no `zero_grad()` and an optimizer step
- **TG004** `DataLoader` starving a CUDA device (`num_workers=0`, no `pin_memory`)
- **TG005** `softmax` before a loss that expects raw logits
- **TG010** projected peak VRAM exceeding the configured `target_gpu`
- Grad-provenance dataflow analysis, so a finding depends on whether a value actually
  carries a graph rather than on what the line looks like
- Autofixes as concrete syntax tree rewrites, preserving formatting and comments
- Terminal, JSON, SARIF and GitHub annotation output
- `# noqa: TG001`, `# torch-preflight: ignore[...]` and `# torch-preflight: skip-file` suppression

### VRAM estimation

- `torch-preflight estimate` projects peak memory from a training script without importing or
  executing it
- Remediation solver reporting which change would make a run fit
- 41 bundled architectures, 23 GPUs, 34 cloud instance types
- Exact profiling of arbitrary models on PyTorch's meta device (`[vram]` extra)
- Hugging Face architecture lookup (`[hub]` extra)
- `VRAMGuard` context manager, failing a run at step 0 rather than OOM at step 400
- Constants calibrated against measured hardware; 5.0% mean absolute error against
  measured peaks

### Packaging

- Python 3.9–3.13
- Base install has no heavy dependencies and never imports torch, asserted in CI
- Pre-commit hook and GitHub Action
