# Rules

| Code | Severity | Category | Problem |
|------|----------|----------|---------|
| **TG001** | error / warning | `CRITICAL_OOM` / `PERFORMANCE_WARN` | A tensor is stored (`losses.append(loss)`, `total += loss`, `cache[k] = out`) with its autograd graph still attached. Error when nothing backwards it, so the activations are retained; warning when backward has already freed them |
| **TG002** | error | `CRITICAL_OOM` | Validation/inference runs a forward pass without `torch.no_grad()` / `inference_mode()` |
| **TG003** | error | `CONVERGENCE_BUG` | `.backward()` runs in a loop with no `zero_grad()` anywhere in it |
| **TG004** | warning | `PERFORMANCE_WARN` | `DataLoader` with `num_workers=0` or no `pin_memory` while the file targets CUDA |
| **TG005** | error | `CONVERGENCE_BUG` | `softmax` applied before `CrossEntropyLoss` (which applies `log_softmax` itself) |
| **TG006** | error | `CONVERGENCE_BUG` | Binary cross-entropy paired with the wrong activation — `sigmoid` before `BCEWithLogitsLoss`, or raw logits into `BCELoss` |
| **TG012** | error | `CONVERGENCE_BUG` | `DataLoader` under DDP with no `DistributedSampler` — every rank trains on identical batches |
| **TG014** | error | `CONVERGENCE_BUG` | Gradient accumulation without dividing the loss — summed gradients scaled as if averaged, equivalent to an N× learning rate |
| **TG011** | error | `CONVERGENCE_BUG` | `model.eval()` in an epoch loop with no matching `train()` — only the first epoch trains properly |
| **TG007** | warning | `PERFORMANCE_WARN` | A GPU sync (`.item()`, `.cpu()`, `.numpy()`) inside a loop nested in the training step, or `cuda.synchronize()` every step |
| **TG008** | warning | `CONVERGENCE_BUG` | A training run whose randomness is unseeded, or seeded for only some of torch / NumPy / `random` |
| **TG013** | warning | `PERFORMANCE_WARN` | A host-to-device transfer repeated every iteration — loop-invariant data, a host factory, or the model itself |
| **TG010** | error | `CRITICAL_OOM` | Projected peak VRAM exceeds the configured `target_gpu` — see [VRAM estimation](vram-estimation.md) |

The numbering has gaps. TG009 is **deliberately not implemented**: in-place operations on
tensors needed for backward already raise a precise error from PyTorch itself, naming the
tensor and its version counter, so a pre-flight check adds nothing you would not learn a
second later. Every rule here targets a failure that is *silent*.

`torch-preflight explain TG001` prints the full write-up for any rule, including why it costs
money and how to fix it.

## The example

`examples/bad_train.py` contains one of each. Running torch-preflight on it:

```
examples/bad_train.py
  42:13  error   TG003 (CONVERGENCE_BUG)
  `.backward()` runs in this loop but nothing calls `zero_grad()`; gradients accumulate across iterations and every step trains on the running sum.
   42 │             loss.backward()          # TG003: nothing calls optimizer.zero_grad()
      │             ^^^^^^^^^^^^^^^
  help: Add `optimizer.zero_grad(set_to_none=True)` at the start of the loop body (or right after `optimizer.step()`).

  48:27  warning TG001 (PERFORMANCE_WARN)
  `losses.append(...)` keeps a graph-attached tensor whose backward has already run. Its activations are freed, so what accumulates is the graph nodes — host memory rather than VRAM, but still growing every iteration, and the stored value stays silently differentiable.
   48 │             losses.append(loss)      # TG001: retains graph nodes (backward already ran)
      │                           ^^^^
  help: Use `.item()` to keep just the scalar value, or `.detach()` to keep the tensor without its graph.
  fix:  add .detach() (run with --fix)
```

`examples/good_train.py` is the same script written correctly; torch-preflight reports
nothing on it, and a test asserts that stays true.

# False positives

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
- **retention a later backward depends on** — a container whose contents reach a backward
  pass is holding those graphs on purpose, as pipeline-parallel schedules do with microbatch
  losses, chunked loss modules do when they accumulate per-chunk losses and return the total,
  and `torch.distributed.autograd` does with a chain of intermediates. TG001 stays quiet
  there, because detaching would break the backward rather than save memory.

  This is decided by reachability, not by syntax. `torch.stack(losses).mean()` is a throwaway
  reduction when it is logged and load-bearing when it becomes the training loss, and the two
  are written identically — so the logging case is still reported and the loss case is not.
  Handing back a bare container (`return losses`) is *not* treated as evidence either way,
  since a caller is as likely to read values out of it as to backward through it

If a rule misfires on real code, that is a bug worth filing.

