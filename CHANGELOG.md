# Changelog

All notable changes to torch-preflight are recorded here. This project follows
[semantic versioning](https://semver.org/).

## [0.1.0] — unreleased

First release.

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
