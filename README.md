# torch-guard

[![CI](https://github.com/highwaterlabs/torch-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/highwaterlabs/torch-guard/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/torch-guard/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-294%20passing-brightgreen)](https://github.com/highwaterlabs/torch-guard/actions/workflows/ci.yml)

[**Docs**](docs/) | [**Rules**](docs/rules.md) | [**VRAM estimation**](docs/vram-estimation.md) | [**CLI**](docs/cli.md)

**A static analyzer for PyTorch that understands autograd** — it catches VRAM leaks and
silent convergence bugs at commit time, and tells you whether your training run will OOM
before you launch it.

```console
$ torch-guard check train.py

train.py
  7:19  error   TG001 (CRITICAL_OOM)
  `losses.append(...)` stores a tensor that is still attached to the autograd graph;
  every iteration's graph is retained in VRAM.
    7 │     losses.append(loss)
      │                   ^^^^
  help: Use `.item()` to keep just the scalar value, or `.detach()` to keep the tensor
        without its graph.
  fix:  add .detach() (run with --fix)

Found 1 error in 1 file(s).
```

<p align="center"><i>One line, one wasted GPU hour. Caught in milliseconds, before it runs.</i></p>

- 🔍 **Six rules for bugs `ruff` and `flake8` cannot see** — retained autograd graphs,
  missing `zero_grad()`, evaluation without `no_grad()`, starved dataloaders, doubled softmax
- 🧮 **Pre-flight VRAM estimation** — projects peak memory from your script and says which
  change would make it fit
- 🛠️ **Autofixes** via concrete syntax tree rewrites, so formatting and comments survive untouched
- 📊 **Measured, not guessed** — every constant calibrated against real hardware, **5.0% mean
  error** versus measured peaks
- 🤫 **Quiet on real code** — **5 findings across PyTorch's own 2,239 files**, all deliberate
- ⚡ **No GPU and no PyTorch required** — pure static analysis over [LibCST](https://github.com/Instagram/LibCST); a CI job asserts torch is never imported
- 🐍 Python 3.9–3.13, `pyproject.toml` config, pre-commit hook, GitHub Action, SARIF output

`ruff` and `flake8` understand Python. They don't understand autograd graphs, gradient
accumulation, or what `num_workers=0` does to eight GPUs waiting on one CPU. torch-guard
is built for the bugs that only cost money once you're paying for a GPU.

## Table of contents

- [Getting started](#getting-started)
- [The line that costs you a GPU hour](#the-line-that-costs-you-a-gpu-hour)
- [Will this fit on the GPU I'm about to rent?](#will-this-fit-on-the-gpu-im-about-to-rent)
- [Why you can trust the numbers](#why-you-can-trust-the-numbers)
- [Integrations](#integrations)
- [Documentation](#documentation)
- [What stays free](#what-stays-free)

## Getting started

```bash
pip install torch-guard
```

```bash
torch-guard check ./src/                        # lint a tree
torch-guard check ./src/ --fix                  # apply the safe fixes
torch-guard estimate train.py --gpu a100-80gb   # will this run fit?
torch-guard explain TG003                       # why a rule exists, and what it costs
```

The base install has no heavy dependencies. `torch-guard[hub]` adds Hugging Face
architecture lookup; `torch-guard[vram]` adds exact meta-device profiling.

## The line that costs you a GPU hour

```python
losses = []
for batch, targets in loader:
    optimizer.zero_grad()
    loss = criterion(model(batch), targets)
    loss.backward()
    optimizer.step()
    losses.append(loss)          # ← keeps every step's graph alive in VRAM
```

You have written this. Everyone has. `loss` still carries its computational graph, so
appending it retains every intermediate activation from that step — and the next, and the
next. Memory climbs linearly until CUDA gives up, hours in.

**Why this is hard:** `losses.append(x)` is only a bug when `x` carries a graph. torch-guard
runs a dataflow pass to find out, tracing values across assignments, arithmetic, tensor
methods and function scopes, and refusing to propagate through `.detach()`, `.item()` or
`argmax`. So `losses.append(loss.item())` stays silent, and so does anything inside
`torch.no_grad()`. A linter that pattern-matched on `.append(` would be unusable.

See [all six rules →](docs/rules.md)

## Will this fit on the GPU I'm about to rent?

```console
$ torch-guard estimate finetune.py --gpu a100-80gb

Model      llama-2-7b  (arch-snapshot)   6.74 B params
Config     amp · AdamW · batch 4 · seq 2048

  weights            25.10 GiB      autocast cache     12.55 GiB
  gradients          25.10 GiB      activations        66.44 GiB
  optimizer state    50.21 GiB      fragmentation      18.84 GiB
  ─────────────────────────────────────────────────────────────
  projected peak    198.37 GiB   (178.53 GiB – 218.21 GiB)

Target     NVIDIA A100 80GB (78.0 GiB usable)   →   254% of capacity   ✗ OOM

What would make it fit:
  ✗  − 66.30 GiB  →  132.07 GiB   gradient checkpointing
  ✗  − 41.61 GiB  →  156.76 GiB   8-bit AdamW (bitsandbytes)
  ✗  −112.02 GiB  →   86.35 GiB   all of the above + flash attention
                                  + halve micro-batch — still does not fit
```

A single 80GB A100 is the wrong tool for a full 7B fine-tune at sequence 2048. Better to
learn that now than after the instance is running.

Model, batch size, sequence length, precision and sharding are read out of your script —
nothing is imported or executed. **41 architectures** ship built in, **23 GPUs** and
**34 cloud instances** are known by name (`--gpu p4de.24xlarge` works), and anything else
is measured exactly on PyTorch's meta device without allocating a byte.

Every other estimator stops at the number. The list of what to *change* is the part you
actually wanted.

See [VRAM estimation →](docs/vram-estimation.md)

## Why you can trust the numbers

Memory estimators are easy to write and easy to be quietly wrong about. So:

|  |  |
|---|---|
| **Constants are measured** | Activation coefficients from `saved_tensors_hooks` on the meta device; allocator behaviour and CUDA context from a real GPU. Measurement showed the published Megatron constants are a midpoint of two regimes — models with dropout retain 3× the attention tensors — so Llama-class models are charged the cheaper rate they actually pay. |
| **Projections are checked** | **5.0% mean absolute error** against measured peaks for GPT-2, BERT and DistilBERT on a T4. Harness and fixtures in [`tests/calibration/`](tests/calibration/), so you can re-run them. |
| **It refuses to guess** | An unrecognised model reports `UNKNOWN` and widens the interval rather than inventing a parameter count. Verdicts are bands with an error range, never a fabricated "95% risk" score. |
| **It stays quiet** | **5 findings across PyTorch's own 2,239 files**, every one deliberate. That pass found four bugs in the rules, now regression-tested. |

294 tests. PyTorch's entire source tree lints in ~50 seconds.

## Integrations

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/highwaterlabs/torch-guard
    rev: v0.1.0
    hooks:
      - id: torch-guard
```

```yaml
# .github/workflows/lint.yml
- uses: highwaterlabs/torch-guard@v0
  with:
    paths: src/
    format: github      # inline PR annotations
```

SARIF output feeds GitHub code scanning; JSON feeds everything else. Set `target_gpu` in
`pyproject.toml` and CI fails on a projected OOM before the job is ever submitted.

See [CI integration →](docs/ci.md)

## Documentation

| | |
|---|---|
| [Rules](docs/rules.md) | All six rules, and the false positives deliberately suppressed |
| [VRAM estimation](docs/vram-estimation.md) | Custom architectures, CI gating, `VRAMGuard`, accuracy |
| [CLI reference](docs/cli.md) | Commands, flags, exit codes, autofixes |
| [Configuration](docs/configuration.md) | `pyproject.toml` and inline suppression |
| [CI integration](docs/ci.md) | GitHub Action, pre-commit, SARIF |
| [Architecture](docs/architecture.md) | How the analysis pipeline works |
| [Development](docs/development.md) | Tests, adding a rule, roadmap |

Design notes live in [`design/`](design/), including the
[RFC](design/rfcs/0001-vram-estimator.md) behind the estimator and the
[spike](design/spikes/0001-meta-device-activation-capture.md) the cost model rests on.

## What stays free

MIT licensed. These are commitments, not just current state:

- **Every rule that has ever shipped free stays free.**
- **The estimator, the remediation solver and `VRAMGuard` stay complete** — not a demo tier.
- **The rule API stays open**, so anyone can write and ship their own rules.
- **The calibration method and data stay public and reproducible.** Numbers are only worth
  trusting if you can check them.

A hosted service may come later for things that genuinely need a server or a team.
Nothing above is part of that.

## Contributing

Issues and pull requests are welcome. Adding a rule is one file plus a `@register`
decorator — see [development](docs/development.md) for the walkthrough and the test
conventions.

## License

MIT — see [LICENSE](LICENSE).
