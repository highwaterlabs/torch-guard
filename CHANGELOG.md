# Changelog

All notable changes to torch-preflight are recorded here. This project follows
[semantic versioning](https://semver.org/).

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
