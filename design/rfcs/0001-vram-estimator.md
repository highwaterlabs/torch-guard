# RFC 0001 — Pre-flight VRAM estimation

**Status:** Implemented — phases 1, 2 and 3 complete
**Created:** 2026-08-12
**Affects:** new `torch_guard.vram` subpackage, new `estimate` command, new rule TG010, packaging

---

## 1. Problem

An engineer is about to launch a training job on a rented A100 or a 4090 on RunPod. The
config is wrong — batch size too large, AdamW in fp32, no gradient checkpointing — and the
job dies with CUDA OOM. Sometimes immediately, sometimes forty minutes in once activations
peak. Either way they have paid for the GPU and lost the wall-clock time.

torch-guard already reads the training script. It should be able to say *before the job
starts* whether the configuration fits the target hardware, and if not, what to change.

## 2. Goals / non-goals

**Goals**
- Project peak VRAM from a training script + target GPU, with an explicit error interval.
- Work in CI with **no GPU** and, in the default install, **no torch**.
- Tell the user what change would make it fit — not just that it will not.
- Degrade honestly: report "unknown" rather than invent a parameter count.

**Non-goals**
- Predicting throughput, step time or cost-per-epoch. Memory only, for now.
- Modelling activation memory of arbitrary user ops exactly in the static tier.
- Replacing an actual run. This is a pre-flight check, not a guarantee.

## 3. Architecture

The expensive dependency is needed for exactly **one input**: the parameter count and
activation footprint of the model. Everything downstream is dependency-free arithmetic.
So both paths converge on one intermediate object.

```
                    ┌─ static provider ─┐   (no deps: CST + bundled arch snapshot)
train.py ──────────►│                   ├──► ModelProfile ──┐
                    └─ meta provider ───┘   (needs torch)    │
                       imports the model, runs a             │
                       meta-device forward/backward          ▼
                                                        cost model  ◄── RunConfig (from CST)
                                                        (zero deps) ◄── Hardware DB
                                                             │
                                                             ▼
                                                   VramReport + remediation
```

```python
@dataclass
class ModelProfile:
    param_count: int
    trainable_params: int
    buffer_bytes: int
    activation_bytes_per_sample: Optional[int]
    source: Literal["arch-snapshot", "hub", "formula", "meta-device"]
    confidence: Confidence            # HIGH | MEDIUM | LOW | UNKNOWN
```

**Why this split.** The CI-gating value (catch a config that will OOM on every PR) needs
only the cost model, which is free. The exactness value (arbitrary custom architectures)
needs torch, and is opt-in. Same command, same output, same rules — only the confidence
label and the `source` field change.

### Proposed layout

```
src/torch_guard/vram/
  hardware.py          # GPU + cloud instance database              no deps
  costmodel.py         # ModelProfile + RunConfig -> VramReport      no deps
  solver.py            # "what change makes it fit"                  no deps
  archdb.py            # bundled snapshot of known architectures     no deps
  extract.py           # RunConfig from the CST (reuses analysis/)   no deps
  providers/
    __init__.py        # registry, tries best-first, degrades
    static.py          # archdb + analytic formulas                  no deps
    hub.py             # live huggingface_hub config lookup          [hub]
    meta.py            # torch meta-device profiling                 [vram]
  guard.py             # VRAMGuard runtime context manager           [vram]
```

## 4. Packaging: one wheel, optional dependencies

`pip install torch-guard` and `pip install torch-guard[vram]` install the **byte-identical
wheel**. All code ships always. An extra adds only *dependencies*.

```toml
[project.optional-dependencies]
hub  = ["huggingface_hub>=0.20"]
vram = ["torch>=2.1"]
all  = ["torch-guard[hub,vram]"]
```

Code that needs an optional dependency imports it **lazily, inside the function**:

```python
def _try_meta(target):
    try:
        from .meta import profile_with_meta_device   # torch imported HERE, not at module top
    except ImportError:
        return None                                  # extra absent; fall through
    return profile_with_meta_device(target)
```

**Invariant:** no module reachable from `import torch_guard` may import torch,
`huggingface_hub`, or anything else outside base dependencies at top level.

**Enforcement** (this is not optional — one careless import silently breaks the light
install, and normal CI will not catch it because CI has everything installed):

```python
def test_base_install_never_imports_torch():
    subprocess.run([sys.executable, "-c",
        "import sys, torch_guard;"
        "torch_guard.check_source('t.py', 'x = 1');"
        "assert 'torch' not in sys.modules"], check=True)
```

plus a CI job that installs the base package only and runs the whole suite.

### Network policy

A linter that hits the network by default is a bad linter — hermetic CI, offline dev,
rate limits. Therefore:

- A **bundled JSON snapshot** of the top few hundred architecture configs ships in the base
  package (tens of KB). Zero network, zero deps, covers the common case.
- The `[hub]` extra enables live lookup only for architectures missing from the snapshot,
  cached on disk.
- Live lookup is **never** triggered by `torch-guard check`. Only by explicit `estimate`.

## 5. Model resolution

Cheapest first, each layer falling through to the next.

**Layer 1 — string literal, zero execution.** `AutoModel.from_pretrained("bert-base-uncased")`
puts the architecture in a string literal the CST can read. Resolve against the bundled
snapshot, then the hub. No import, no torch, no user code executed. Covers a large share of
real scripts.

**Layer 2 — local class, resolvable args.** For `model = Classifier(num_classes=10)`,
`provenance._is_model_expr` already locates model construction sites. Constant-fold the
arguments from the CST, import the **defining module** (never the training script), and
re-evaluate that single call under `torch.device("meta")`. Requires `[vram]`.

**Layer 3 — honest failure.** When an argument cannot be folded, do not guess. Report what
blocked resolution:

```
Could not resolve the model automatically.
  examples/train.py:31  Classifier(num_classes=cfg.num_classes)
                                               ^^^^^^^^^^^^^^^ set at runtime
  Pass it explicitly:  --model examples.model:Classifier --model-args num_classes=10
```

**Explicitly rejected:** importing `train.py` itself and scanning globals for an
`nn.Module`. Model construction almost always happens inside `main()`, so globals hold the
class and not the instance — and importing an unguarded training script starts training.

Phase 1 ships Layers 1 and 3 plus explicit `--model`. Layer 2 is phase 2.

## 6. Cost model

```
peak = weights + gradients + optimizer_state + activations + cuda_context + fragmentation
```

| Term | Model |
|---|---|
| weights | `P × bytes(param_dtype)` |
| gradients | `P_trainable × bytes(grad_dtype)` |
| optimizer | SGD `0` · SGD+momentum `1×P×4` · Adam/AdamW `2×P×4` · 8-bit Adam `2×P×1` · AMP adds an fp32 master copy `+4×P` |
| activations | transformer: Megatron-LM formula, per layer ≈ `s·b·h·(34 + 5·a·s/h)` bytes at fp16; CNN: sum of feature-map tensors from the arch snapshot |
| cuda context | several hundred MB per process, fixed, easy to forget |
| fragmentation | multiplier on the above, allocator-dependent |

Modifiers: gradient checkpointing (large activation reduction), flash attention (removes
the `O(s²)` attention-matrix term), ZeRO/FSDP sharding (divides gradients and/or optimizer
state by world size), gradient accumulation (does **not** reduce activation peak — a common
misconception worth encoding).

**Every empirical constant in this table is a calibration target, not a truth.** See §9.

## 7. Risk reporting

The original sketch proposed a "95% failure risk" score. We have no data to calibrate a
probability against, so that number would be fabricated. Instead: a utilization ratio and
an error interval, banded.

| Band | Condition |
|---|---|
| `FITS` | upper bound of interval < usable VRAM |
| `TIGHT` | estimate < usable, upper bound ≥ usable |
| `LIKELY_OOM` | estimate ≥ usable, lower bound < usable |
| `CERTAIN_OOM` | lower bound ≥ usable |
| `UNKNOWN` | model could not be resolved |

Hardware DB records **usable** VRAM, not advertised (a 24GB 4090 is ~23.6GB, less with a
display attached), and maps cloud instance names (`p4d.24xlarge`, `p5.48xlarge`, RunPod
SKUs) to their GPUs.

### Remediation solver

The differentiator. Search {checkpointing, optimizer choice, precision, batch size,
sharding} for combinations that fit, and rank by disruptiveness:

```
Target  A100 80GB (79.2 GB usable)   →   127% of capacity   ✗ OOM

What would make it fit:
  8-bit AdamW (bitsandbytes)      −40.4 GB  →  60.1 GB  ✓
  gradient checkpointing           −6.9 GB  →  93.6 GB  ✗
  FSDP across 2× A100             −50.2 GB  →  50.3 GB  ✓
```

Pure arithmetic on the cost model, so it ships in the free tier.

## 8. Surface area

```bash
torch-guard estimate train.py --gpu rtx4090
torch-guard estimate --model mypkg.models:build_gpt --batch-size 8 --seq-len 4096
torch-guard gpus
```

Every extracted field is overridable: `--batch-size --seq-len --dtype --optimizer
--world-size --checkpointing`.

**TG010** turns it into a CI gate through the existing `check` path:

```toml
[tool.torch-guard]
target_gpu = "rtx4090"     # check now fails when a confident estimate exceeds this
```

TG010 fires only at `HIGH`/`MEDIUM` confidence. An `UNKNOWN` profile must never fail a build.

## 9. Accuracy and calibration

This feature dies on a wrong number. A confident wrong OOM prediction is worse than none.

- `tests/calibration/` holds known `(model, config) → measured peak` pairs from published
  figures and real runs.
- Tests assert estimates land within a declared tolerance per tier.
- Tolerances are checked in and tightened deliberately; loosening one requires a reason in
  the commit message.

Without this the estimator is vibes. With it, accuracy is a tracked regression surface.

## 10. Phasing

**Phase 1 — static tier, no new required deps.** hardware DB · cost model · arch snapshot ·
RunConfig extraction · solver · `estimate` command · TG010 · calibration harness · base-install
CI job. Ships the whole CI-gating story.

**Phase 2 — exact tier.** Spike [0001](../spikes/0001-meta-device-activation-capture.md) first,
then `providers/meta.py`, `[vram]` extra, model autodetection Layer 2.

**Phase 3 — runtime.** `VRAMGuard` context manager, verification against
`torch.cuda.max_memory_allocated()`, feeding measurements back into calibration.

Static first is deliberate: if the exact provider went first it would feed the cost model
perfect inputs, hiding errors in the analytic formulas that the free tier depends on.

## 11. Decisions taken

| # | Decision | Rationale |
|---|---|---|
| 1 | Meta device is phase 2 | Needs a spike; static tier delivers CI value first |
| 2 | Explicit `--model` in phase 1, autodetect in phase 2 | Explicit is predictable; auto-detection is magic that breaks |
| 3 | Wire up `huggingface_hub` | Broad coverage — but bundled snapshot first, network never in `check` |
| 4 | Build static tier first | Forces the shared cost model to be correct before it can hide behind exact inputs |

## 12. Open questions

- Snapshot contents and refresh process — how do we regenerate it, and how often?
- Does `estimate` need a machine-readable format on day one, or is terminal enough?
- Multi-GPU: model rank-0 only, or the full world including uneven pipeline stages?
