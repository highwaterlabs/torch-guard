# torch-guard

**A static analyzer for PyTorch training code.** It finds VRAM leaks, silent convergence
bugs and GPU pipeline stalls at commit time — and tells you whether your training run will
OOM *before* you rent the A100.

```bash
pip install torch-guard
torch-guard check ./src/
torch-guard estimate train.py --gpu a100-80gb
```

`ruff` and `flake8` understand Python. They do not understand autograd graphs, gradient
accumulation, or what `num_workers=0` does to eight GPUs waiting on one CPU. torch-guard
does, and it needs neither a GPU nor a PyTorch install to run — it is pure static analysis
over [LibCST](https://github.com/Instagram/LibCST).

---

## What it catches

| Code | Severity | Category | Problem |
|------|----------|----------|---------|
| **TG001** | error | `CRITICAL_OOM` | A tensor is stored (`losses.append(loss)`, `total += loss`, `cache[k] = out`) with its autograd graph still attached |
| **TG002** | error | `CRITICAL_OOM` | Validation/inference runs a forward pass without `torch.no_grad()` / `inference_mode()` |
| **TG003** | error | `CONVERGENCE_BUG` | `.backward()` runs in a loop with no `zero_grad()` anywhere in it |
| **TG004** | warning | `PERFORMANCE_WARN` | `DataLoader` with `num_workers=0` or no `pin_memory` while the file targets CUDA |
| **TG005** | error | `CONVERGENCE_BUG` | `softmax` applied before `CrossEntropyLoss` (which applies `log_softmax` itself) |
| **TG010** | error | `CRITICAL_OOM` | Projected peak VRAM exceeds the configured `target_gpu` — see [below](#pre-flight-vram-estimation) |

`torch-guard explain TG001` prints the full write-up for any rule, including why it costs
money and how to fix it.

### The example

`examples/bad_train.py` contains one of each. Running torch-guard on it:

```
examples/bad_train.py
  42:13  error   TG003 (CONVERGENCE_BUG)
  `.backward()` runs in this loop but nothing calls `zero_grad()`; gradients accumulate across iterations and every step trains on the running sum.
   42 │             loss.backward()          # TG003: nothing calls optimizer.zero_grad()
      │             ^^^^^^^^^^^^^^^
  help: Add `optimizer.zero_grad(set_to_none=True)` at the start of the loop body (or right after `optimizer.step()`).

  45:27  error   TG001 (CRITICAL_OOM)
  `losses.append(...)` stores a tensor that is still attached to the autograd graph; every iteration's graph is retained in VRAM.
   45 │             losses.append(loss)      # TG001: keeps the whole graph alive
      │                           ^^^^
  help: Use `.item()` to keep just the scalar value, or `.detach()` to keep the tensor without its graph.
  fix:  add .detach() (run with --fix)
```

`examples/good_train.py` is the same script written correctly; torch-guard reports
nothing on it, and a test asserts that stays true.

## How it works

```
source.py
   │
   ├─ LibCST parse ──────────────► concrete syntax tree (comments + formatting preserved)
   │
   ├─ FileContext ───────────────► imports, CUDA usage, Lightning detection, softmax vars
   │
   ├─ Provenance analysis ───────► which names carry a live autograd graph?
   │      seeds: .backward() receivers, criterion/model call results, requires_grad=True
   │      propagate to a fixpoint across assignments
   │      never propagate through .detach() / .item() / float()
   │
   ├─ Rules (scope + loop + no_grad aware) ──► diagnostics [+ optional fix]
   │
   └─ Reporters ─────────────────► terminal · JSON · GitHub annotations · SARIF
```

The provenance pass is the part a syntax-only linter cannot replicate. `losses.append(x)`
is only a leak if `x` is grad-bearing, so torch-guard tracks where `x` came from —
across assignments, arithmetic, tensor methods and function scopes — and stays quiet when
the value was detached or the code is already inside `torch.no_grad()`.

## Pre-flight VRAM estimation

Before renting the GPU, ask whether the run fits:

```bash
$ torch-guard estimate finetune.py --gpu a100-80gb

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
torch-guard estimate train.py --gpu rtx4090 --batch-size 1 --seq-len 512 --dtype pure-bf16
torch-guard estimate --model llama-2-7b --gpu 8xa100-80gb --sharding zero-3
torch-guard estimate --params 13B --gpu-memory 48GiB
torch-guard gpus --instances        # every known GPU and cloud instance
```

Cloud instance names work directly: `--gpu p4de.24xlarge`, `--gpu ml.p5.48xlarge`,
`--gpu a2-ultragpu-8g`.

### Custom architectures

Models outside the bundled snapshot can be measured exactly, with `pip install
"torch-guard[vram]"`:

```bash
torch-guard estimate --model mypkg.models:build_gpt \
    --model-args layers=24 --model-args hidden=1024 \
    --gpu a100-80gb --batch-size 8 --seq-len 1024 --dtype amp
```

The model is instantiated on PyTorch's `meta` device — **zero bytes allocated, no GPU
required** — so parameter counts are exact rather than estimated, and a forward pass under
`saved_tensors_hooks` captures precisely the tensors autograd retains for backward. That
is activation memory by definition, not by formula.

This path imports and executes your code, so it is opt-in and explicit. `torch-guard check`
never reaches it.

### As a CI gate

Declare the hardware your team trains on and TG010 fails the build when a config will not
fit it:

```toml
[tool.torch-guard]
target_gpu = "rtx4090"
```

TG010 is deliberately conservative. It stays silent unless `target_gpu` is set, it needs a
model it can identify, it **never fires on a low-confidence estimate**, and it never
touches the network.

### How accurate is it?

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
retain one — so torch-guard charges Llama-class models the cheaper rate they actually pay.

The allocator constants are measured on a real GPU, and end-to-end projections are checked
against measured peaks from actual training steps:

| | measured | estimated | error |
|---|---|---|---|
| GPT-2, batch 4 × seq 128 | 2.97 GiB | 2.94 GiB | −1.3% |
| GPT-2, batch 8 × seq 256 | 5.81 GiB | 4.66 GiB | −19.8% |
| BERT-base, batch 4 × seq 128 | 2.45 GiB | 2.39 GiB | −2.4% |
| BERT-base, batch 8 × seq 256 | 3.39 GiB | 3.33 GiB | −2.0% |
| DistilBERT, batch 4 × seq 128 | 1.54 GiB | 1.48 GiB | −3.9% |
| DistilBERT, batch 8 × seq 256 | 1.96 GiB | 1.94 GiB | −0.8% |

Mean absolute error 5.0%, on a Tesla T4. Regenerate with
`tests/calibration/measure_cuda.py --models`.

Every estimate carries an interval, and the verdict bands account for it — there is no
fabricated "95% failure risk" probability, because there is no data to calibrate one
against. If the model cannot be identified, torch-guard reports `UNKNOWN` rather than
guessing a parameter count.

**Known gaps**, tracked in [design/TODO.md](design/TODO.md):

- CNN activation memory is not modelled — vision entries carry parameter counts only, and
  the report says so and widens the interval instead of inventing a number.
- Causal-LM loss temporaries are modelled as a **floor**. GPT-2 at batch 8 × seq 256 still
  exceeds the estimate by ~20%, so real loss implementations keep copies the model does not
  capture. Encoder models land within 4%.
- Entry-point profiling measures the forward pass only, so the transient where a
  checkpointed layer is recomputed during backward is still modelled analytically.
- Calibration covers one GPU (T4) and three model families. `CUDA_CONTEXT_BYTES` in
  particular is a single data point; larger cards plausibly differ, and
  `hardware.Gpu.context_mib` exists to hold per-card numbers as they arrive.
- Encoder-decoder models (T5, Whisper) have parameter counts but no activation model.

## Install

```bash
pip install torch-guard              # linter + static VRAM estimation, no torch needed
pip install "torch-guard[hub]"       # + look up unknown architectures on the HF hub
pip install "torch-guard[vram]"      # + exact meta-device profiling (phase 2)
```

The base install has no heavy dependencies and never imports torch — a CI job asserts it.
Extras add *dependencies only*; the wheel is byte-identical either way.

### Guarding a run at runtime

The estimator answers "will this fit?" before the job is submitted. `VRAMGuard` answers it
*inside* the process, against the model that actually exists:

```python
from torch_guard import VRAMGuard

with VRAMGuard(model, optimizer=optimizer, batch_size=32, seq_len=2048):
    train()
```

```
VramRiskError: torch-guard: this configuration is projected to need 701 MiB
(456 MiB-946 MiB) on limit 256MiB, which has 0.2 GiB usable. Verdict: CERTAIN_OOM.
  breakdown: weights 128 MiB, gradients 128 MiB, optimizer state 256 MiB, ...
  smallest change that fits: 8-bit AdamW (bitsandbytes) + pure bf16 weights
```

Parameters, gradients, optimizer state and the autocast cache come from the live model and
are exact; the optimizer kind and precision are read off the objects you pass. It **raises
only when the run cannot fit even at the optimistic end of the interval** — anything less
certain is a warning, because aborting a training job on a guess is worse than the OOM it
was trying to prevent. `strict=True` opts into raising on likely failures too.

Needs `pip install "torch-guard[vram]"`. On exit, `guard.measured_peak` and
`guard.accuracy` compare the projection against what the run actually used.

## Usage

```bash
torch-guard check ./src/              # check a tree
torch-guard check train.py --fix      # apply autofixes in place
torch-guard check train.py --diff     # show what --fix would do
torch-guard check . -f json           # machine-readable
torch-guard check . -f sarif          # GitHub code scanning
torch-guard check . -f github         # inline PR annotations
torch-guard rules                     # list rules
torch-guard explain TG003             # full write-up for one rule
```

`torch-guard ./src` is shorthand for `torch-guard check ./src`.

**Speed.** PyTorch's own source — 2285 files — takes about 50 seconds on an 8-core laptop.
Checking runs across processes by default; `--jobs 1` disables that, and `--fix` always
runs sequentially.

**Exit codes:** `0` clean · `1` findings at or above `--fail-on` (default `error`, so
performance warnings do not break CI) · `2` bad invocation.

### Autofixes

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

## Configuration

`pyproject.toml`:

```toml
[tool.torch-guard]
select = ["TG001", "TG002", "TG003", "TG005"]   # omit to enable everything
ignore = ["TG004"]
exclude = ["tests/fixtures", "notebooks"]
fail_on = "warning"                              # default: "error"

[tool.torch-guard.severity]
TG004 = "note"                                   # downgrade instead of disabling
```

A standalone `.torch-guard.toml` (same keys, no `[tool.torch-guard]` header) also works.
torch-guard walks up from the first checked path until it finds one.

### Suppressing findings

```python
losses.append(loss)  # noqa: TG001
losses.append(loss)  # torch-guard: ignore[TG001]
losses.append(loss)  # noqa
```

`# torch-guard: skip-file` anywhere in a file skips it entirely.

## CI integration

### GitHub Action

```yaml
- uses: highwaterlabs/torch-guard@v0
  with:
    paths: src/
    format: github        # inline PR annotations
```

Or with code scanning:

```yaml
- uses: highwaterlabs/torch-guard@v0
  with:
    paths: src/
    format: sarif
    output: torch-guard.sarif
- uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: torch-guard.sarif
```

### Pre-commit

```yaml
repos:
  - repo: https://github.com/highwaterlabs/torch-guard
    rev: v0.1.0
    hooks:
      - id: torch-guard
```

## False positives

The provenance analysis is deliberately coarse — flow-insensitive within a scope, and it
leans on naming conventions (`loss`, `criterion`, `model`, `val_loader`) that PyTorch code
follows almost universally. That trade buys speed and zero runtime dependencies at the
cost of occasional noise. Known suppressions already built in:

- code inside `torch.no_grad()` / `inference_mode()` / `set_grad_enabled(False)`
- values that carry no graph to begin with — `argmax`, `argsort`, comparisons, integer
  casts, shape queries — so `preds.append(logits.argmax(-1))` stays quiet while
  `losses.append(loss.sum())` does not
- Lightning `validation_step` / `test_step` / `predict_step`, and `zero_grad` checks in any
  `LightningModule` (the framework handles both)
- `zero_grad()` inside an `if` in the loop — deliberate gradient accumulation
- pytest's `test_*` functions, which legitimately exercise autograd
- containers created fresh inside the loop body, which cannot accumulate

If a rule misfires on real code, that is a bug worth filing.

## Roadmap

The rule engine is designed so a new check is one file plus a `@register` decorator.
Next up:

- **TG006** `nn.BCELoss` on raw logits (should be `BCEWithLogitsLoss`)
- **TG007** CPU↔GPU thrashing (`.cpu().numpy()`) inside the training loop
- **TG008** non-reproducible runs — `torch.rand` without seeding `torch`, `numpy` and `random`
- **TG009** in-place ops on tensors needed for the backward pass
- **Meta-device profiling** — exact parameter and activation measurement for arbitrary
  custom models via PyTorch's `meta` device, as the `[vram]` extra ([RFC 0001](design/rfcs/0001-vram-estimator.md)
  phase 2, gated on [spike 0001](design/spikes/0001-meta-device-activation-capture.md))
- **Model autodetection** — resolve locally defined model classes by constant-folding their
  constructor arguments

## What stays free

torch-guard is open source under MIT, and these are commitments rather than current state:

- **Every rule that has ever shipped free stays free.** Rules are the reason to install
  this; they will not move behind a paywall.
- **The estimator, the remediation solver and `VRAMGuard` stay free and stay complete.**
  Not a demo tier — if a team uses nothing but the free package, they should be well served.
- **The rule API stays open**, so anyone can write, use and distribute their own rules
  without asking.
- **The calibration method and its seed data stay public and reproducible.** The numbers
  this tool prints are only worth trusting if you can check them, and
  [`tests/calibration/`](tests/calibration/) is how.

A hosted service may exist later for things that genuinely need a server or a team —
cross-repo dashboards, org-wide policy, history. Nothing listed above is part of that.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Adding a rule: create `src/torch_guard/rules/tg0NN_name.py`, subclass `Rule`, decorate with
`@register`, import it in `rules/__init__.py`, then add cases to `tests/test_rules.py` —
at least one that must fire and one that must stay quiet.

## License

MIT
