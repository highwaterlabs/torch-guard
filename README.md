# torch-guard

**Catch PyTorch training bugs before they cost you GPU hours.** VRAM leaks, silent
convergence bugs and pipeline stalls at commit time — plus whether your run will OOM
*before* you rent the A100.

```bash
pip install torch-guard

torch-guard check ./src/                          # lint a tree
torch-guard estimate train.py --gpu a100-80gb     # will this run fit?
```

The base install pulls in no heavy dependencies and never imports torch — a CI job
asserts it. Two optional extras add capability without changing the wheel:
`torch-guard[hub]` looks up unknown architectures on the Hugging Face hub, and
`torch-guard[vram]` measures arbitrary models exactly on PyTorch's meta device.

`ruff` and `flake8` understand Python. They don't understand autograd graphs, gradient
accumulation, or what `num_workers=0` does to eight GPUs waiting on one CPU. torch-guard
does — and it needs neither a GPU nor a PyTorch install, because it's static analysis over
[LibCST](https://github.com/Instagram/LibCST).

## What it catches

| | Problem |
|---|---|
| **TG001** | A tensor stored with its autograd graph still attached — `losses.append(loss)` keeps every step's graph in VRAM |
| **TG002** | Validation running a forward pass without `torch.no_grad()` |
| **TG003** | `.backward()` in a loop with no `zero_grad()` — silently trains on accumulated gradients |
| **TG004** | `DataLoader` starving a CUDA device (`num_workers=0`, no `pin_memory`) |
| **TG005** | `softmax` before `CrossEntropyLoss`, which applies `log_softmax` itself |
| **TG010** | Projected peak VRAM exceeds the GPU you configured |

```
examples/bad_train.py
  45:27  error   TG001 (CRITICAL_OOM)
  `losses.append(...)` stores a tensor that is still attached to the autograd graph;
  every iteration's graph is retained in VRAM.
     45 │             losses.append(loss)
        │                           ^^^^
  help: Use `.item()` to keep just the scalar value, or `.detach()` to keep the tensor
        without its graph.
  fix:  add .detach() (run with --fix)
```

`torch-guard explain TG001` prints the full write-up for any rule. `--fix` applies the
safe fixes; formatting and comments round-trip untouched.

## Will it fit?

```
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

A single 80GB A100 is genuinely the wrong tool for a full 7B fine-tune at sequence 2048.
The point is learning that before you rent one.

The model, batch size, sequence length, precision and sharding are read out of your
script — nothing is imported or executed. Cloud instance names work directly
(`--gpu p4de.24xlarge`).

## Documentation

| | |
|---|---|
| [Rules](docs/rules.md) | Every rule, and the false positives we suppress |
| [VRAM estimation](docs/vram-estimation.md) | Custom architectures, CI gating, `VRAMGuard`, accuracy |
| [CLI reference](docs/cli.md) | Commands, flags, exit codes, autofixes |
| [Configuration](docs/configuration.md) | `pyproject.toml` and inline suppression |
| [CI integration](docs/ci.md) | GitHub Action, pre-commit, SARIF |
| [Architecture](docs/architecture.md) | How the analysis works |
| [Development](docs/development.md) | Tests, adding a rule, roadmap |

## What stays free

torch-guard is MIT licensed, and these are commitments rather than current state:

- **Every rule that has ever shipped free stays free.**
- **The estimator, the remediation solver and `VRAMGuard` stay complete** — not a demo tier.
- **The rule API stays open**, so anyone can write and distribute their own rules.
- **The calibration method and its seed data stay public and reproducible** — the numbers
  are only worth trusting if you can check them, and
  [`tests/calibration/`](tests/calibration/) is how.

A hosted service may exist later for things that genuinely need a server or a team.
Nothing above is part of that.

## License

MIT
