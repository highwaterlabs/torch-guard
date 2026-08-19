# CLI reference

```bash
torch-preflight check ./src/              # check a tree
torch-preflight check train.py --fix      # apply autofixes in place
torch-preflight check train.py --diff     # show what --fix would do
torch-preflight check . -f json           # machine-readable
torch-preflight check . -f sarif          # GitHub code scanning
torch-preflight check . -f github         # inline PR annotations
torch-preflight rules                     # list rules
torch-preflight explain TG003             # full write-up for one rule
```

`torch-preflight ./src` is shorthand for `torch-preflight check ./src`.

## Commands

### `check`

Analyze files or directories for PyTorch anti-patterns.

| Flag | Argument | Default | Description |
|---|---|---|---|
| `paths` | files or directories | `.` | Files or directories to analyze |
| `-f`, `--format` | `terminal`, `json`, `github`, `sarif` | `terminal` | Output format |
| `--fix` | — | `false` | Apply autofixes in place |
| `--diff` | — | `false` | Show what `--fix` would change without writing |
| `--select` | rule code | none | Only run these rules |
| `--ignore` | rule code | none | Skip these rules |
| `--exclude` | path pattern | none | Skip paths matching the pattern |
| `--fail-on` | severity | `error` | Minimum severity that makes the run fail |
| `-j`, `--jobs` | number | CPU count | Number of worker processes; `1` disables parallelism |
| `--target-gpu` | GPU | none | Target GPU for the TG010 projected-OOM gate |
| `--config` | path | none | Path to a config file |
| `--no-color` | — | `false` | Disable coloured output |
| `-q`, `--quiet` | — | `false` | Only print the summary |

### `explain`

Show the full write-up for a rule.

| Argument | Description |
|---|---|
| `code` | Rule code, e.g. `TG001` |

### `rules`

List all available rules.

This command has no additional flags or arguments.

### `estimate`

Project peak VRAM for a training script before you launch it.

| Flag | Argument | Default | Description |
|---|---|---|---|
| `path` | training script | none | Training script to read the config from |
| `--gpu` | GPU or cloud instance | none | Target GPU or cloud instance |
| `--gpu-memory` | capacity | none | Explicit capacity for unlisted hardware, e.g. `48GiB` |
| `--model` | model name or entry point | none | Architecture name or entry point |
| `--model-args` | `KEY=VALUE` | none | Constructor arguments for a `--model` entry point |
| `--params` | parameter count | none | Parameter count when the model is unknown, e.g. `7B` |
| `--online` | — | `false` | Allow Hugging Face Hub lookup for unknown architectures |
| `--batch-size` | number | none | Override per-device micro-batch |
| `--seq-len` | number | none | Override sequence length |
| `--image-size` | number | none | Override image resolution |
| `--dtype` | precision mode | none | Override precision mode |
| `--optimizer` | optimizer | none | Override optimizer |
| `--world-size` | number | none | Number of ranks |
| `--sharding` | sharding strategy | none | Override sharding strategy |
| `--checkpointing` | — | `false` | Assume gradient checkpointing is on |
| `--flash` | — | `false` | Assume Flash Attention / SDPA is on |
| `--inference` | — | `false` | Inference only, with no backward pass |
| `--generate` | — | `false` | Assume autoregressive decoding with a KV cache |
| `--max-context` | number | none | Total tokens the KV cache must hold |
| `-f`, `--format` | `terminal`, `json` | `terminal` | Output format |

### `gpus`

List known GPUs and cloud instances.

| Flag | Argument | Default | Description |
|---|---|---|---|
| `--instances` | — | `false` | Show cloud instances too |

**Speed.** PyTorch's own source — 2285 files — takes about 50 seconds on an 8-core laptop.
Checking runs across processes by default; `--jobs 1` disables that, and `--fix` always
runs sequentially.

**Exit codes:** `0` clean · `1` findings at or above `--fail-on` (default `error`, so
performance warnings do not break CI) · `2` bad invocation.

## Autofixes

Only fixes that cannot change semantics are offered:

| Rule | Fix |
|------|-----|
| TG001 | append `.detach()` to the stored expression |
| TG002 | add `@torch.no_grad()` — **only** to functions whose name says they evaluate, never to a training routine that happens to contain a validation loop |
| TG004 | add `pin_memory=True` |
| TG005 | unwrap the redundant `softmax(...)` call |

TG003 has no autofix: where `zero_grad()` belongs depends on whether you meant to
accumulate gradients, and guessing would be worse than a comment.

Because fixes are LibCST node replacements, everything untouched round-trips byte for
byte — no reformatting, no lost comments.

