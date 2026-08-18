# RFC 0003 — What severity means, and what should fail a build

**Status:** Accepted — implemented
**Created:** 2026-08-18
**Affects:** `Severity` semantics, TG001, TG004, `fail_on` default, config surface

---

## 1. Problem

TG001 now reports two different findings. Storing a graph-attached tensor that **nothing
backwards** retains the activations; storing one **after its backward** retains only the
graph nodes, because `backward()` frees each node's saved tensors as it traverses. The first
is reported as `error` / `CRITICAL_OOM`, the second as `warning` / `PERFORMANCE_WARN`.

The default gate is `fail_on = error`. So the single most-cited line the tool exists to
catch —

```python
loss.backward()
optimizer.step()
losses.append(loss)
```

— no longer fails a build. That is either a correction or a regression, and this RFC decides
which.

The question turned out to be the smaller half of a structural one: **`warning` currently
means two incompatible things**, so no setting of `fail_on` is right for everybody.

## 2. What we measured

Two harnesses, both in `tests/calibration/measure_retention.py`. Activations are counted by
walking the live autograd graph and summing each node's `_saved_*` tensors — deterministic,
byte-identical across runs. Host cost is RSS growth over 20,000 iterations in isolated
processes, median of five.

**Activations retained, per iteration:**

| model | already backwarded | never backwarded |
|---|---|---|
| 9-layer MLP, d=32, batch 4 | 0.0 KiB | 8 KiB |
| 4-layer Transformer, d=256, batch 8 × seq 128 | 0.0 KiB | **190,946 KiB (186 MiB)** |

**Host memory retained, per iteration:** 11–12 KiB (MLP), 12–16 KiB (Transformer).

Three things follow, and the third is the one that decides this RFC.

1. **`backward()` really does free the activations.** Zero, not "less". PyTorch's own error
   confirms it independently: a second `backward()` refuses with *"Trying to backward through
   the graph a second time (or directly access saved tensors after they have already been
   freed)"*.
2. **The backwarded case is not free.** ~15 KiB/iteration of host memory is unbounded linear
   growth — **1.4 GiB per 100k steps**. It will not kill a 5,000-step fine-tune; it is a
   genuine problem for a long pretraining run.
3. **The two costs scale differently.** Activation retention scales with model and batch size
   — 8 KiB/iter on a toy MLP, 186 MiB/iter on a *small* transformer, a 23,000× spread. Node
   retention stays roughly flat across both. So the gap between the two findings widens with
   model size: on the transformer it is **~12,000×**, and 186 MiB/iteration OOMs an 80 GB
   A100 in about **440 steps**.

A correction worth recording, since it nearly produced the wrong answer here: the first pass
at this used RSS alone, which spread over 5.8–20.9 KiB/iteration across identical runs, and
the conclusion "a small leak" was written on top of that noise. A later attempt extrapolated
host cost from node counts (910 nodes for a 12-layer transformer × a per-node constant) and
predicted 280 KiB/iteration — the direct measurement says 12–16. **Node count does not
predict host cost.** Only the directly measured figures above are load-bearing.

## 3. What severity should mean

Today `Severity` is documented as "how loudly a finding should be reported" and `Category` as
"what kind of damage". In practice severity is doing two jobs — *how bad* and *how sure* —
and rules have been assigned by feel. The result:

| | today |
|---|---|
| `error` | TG001 (no backward), TG002, TG003, TG005, TG006, TG010, TG011, TG012, TG014 |
| `warning` | TG001 (backwarded), TG004, TG007, TG008, TG013 |
| `note` | *(unused)* |

`warning` contains both "you have an unbounded memory leak" and "you did not pass
`pin_memory=True`". Those cannot share a gate. In the seven-repo scan, **TG004 alone was 207
of 318 findings** — 13% of every file — so anyone who sets `fail_on = warning` to catch the
TG001 case immediately fails on tutorial `DataLoader` defaults instead.

Proposed definitions, on one axis — *what happens if you ship this*:

- **`error`** — the run produces a wrong result, or dies. Convergence bugs and OOM. Always
  worth failing a build.
- **`warning`** — a real defect with a bounded blast radius: it wastes memory or time, it is
  never intentional, and the fix is unambiguous. Worth failing a build in a repo whose main
  product is training runs.
- **`note`** — a tuning observation. The code is correct; a different default would be
  faster on some hardware. Never worth failing a build.

The distinguishing question for `warning` vs `note` is **"is this code defective, or merely
untuned?"** `losses.append(loss)` is defective — nobody wants the retention, and `.detach()`
costs nothing. `num_workers=0` is a *choice*, and often a deliberate one: it is the portable
default, it is what tutorials use so they run under Windows spawn and in notebooks, and the
right value depends on the host's core count.

## 4. Proposal

1. **Keep the TG001 severity split.** ~12,000× on a realistic model, device memory versus
   host memory, and 440 steps versus 100,000 are not the same finding, and one message cannot
   honestly describe both. This is the split doing its job.

2. **Move TG004 to `note`.** It is the only rule that is purely a tuning observation, and it
   is 65% of all findings on real repos. This is what makes `warning` mean something.

3. **Keep `fail_on = error` as the default**, and document `fail_on = warning` as the
   recommended setting for repositories whose product is training runs. Once (2) lands, that
   setting is usable: it catches the retained-graph leak, the per-element device syncs and the
   unseeded runs, without drowning in `DataLoader` defaults.

4. **Per-rule severity overrides are the general escape hatch** — and they already exist.
   This RFC proposed adding them before checking; `Config.severity_overrides` has been wired
   since the config module was written, applied in `engine.py`, and `docs/configuration.md`
   used `TG004 = "note"` as its worked example. The change is that this becomes the default
   rather than something each user discovers.

   ```toml
   [tool.torch-preflight.severity]
   TG001 = "error"    # we run 500k-step jobs; the host leak is fatal for us
   TG008 = "note"     # we seed in the launcher, not the script
   ```

   So nothing is built here, but it carries the argument: we do not have to be right about
   every rule's default for the tool to be usable, which is what makes (2) safe to ship.

## 5. Consequences

- `torch-preflight check` on the classic append-after-backward line exits 0 by default and
  1 under `fail_on = warning`. That is the intended outcome of this RFC, not a side effect.
- Findings-per-file drops sharply in default reporting once TG004 is a note, which changes
  the headline number in the README and in any future scan write-up. The rules doc must say
  plainly that notes are not counted toward the gate.
- The action's `fail-on` input already accepts `warning`, so no change is needed there. The
  documented CI snippet should be revisited: recommending `fail-on: warning` in the README
  is the natural companion to (2), and it should not ship before it.
- Exit-code behaviour is part of our public contract with CI. This is a breaking change for
  anyone relying on TG001 to fail their build, so it belongs in a minor version with a
  release note that says so, not in a patch.

## 6. Alternatives considered

**Report both TG001 cases as `error`.** Restores the previous CI behaviour with one line.
Rejected: it makes `error` mean "somewhere between 440 steps and never", and we would be
failing builds over 1.4 GiB per 100k steps using a message that cannot claim CUDA OOM. The
measurement is the whole reason the split exists; discarding it to preserve a habit is the
wrong trade.

**Change the default `fail_on` to `warning` without reclassifying TG004.** Rejected on the
scan data: 207 of 318 findings would become build failures, most of them on files where
`num_workers=0` is deliberate. The first thing every user would do is turn the rule off, and
we would have taught them to distrust the gate.

**Add a `confidence` axis alongside severity.** Already parked in `IDEAS.md`, and a real
idea — but it does not apply here. We are not uncertain about either TG001 case; we are
certain about both and they differ in *impact*. Severity is already the impact axis. Adding a
second axis to avoid defining the first one properly would make the tool harder to explain
and leave the TG004 problem untouched.

**Make severity depend on the loop's iteration count.** The honest form of the objection —
the host leak is fatal at 1M steps and irrelevant at 500. Rejected: we cannot know the step
count statically (it is an argument, a config file, a scheduler decision), and guessing it
would put a number we invented behind a build failure. Per-rule overrides (4) let the user
supply the fact we cannot derive, which is the same principle as `target_gpu` in RFC 0001.

## 7. Resolved questions

**TG007 and TG013 stay warnings.** Both are performance findings like TG004, but they fail
the untuned-versus-defective test in the other direction: `preds[i].item()` inside a nested
loop and a host-to-device copy repeated every iteration are *specific defects with specific
fixes*, and nobody chooses them. TG004 reports an unset default whose right value depends on
the machine. The line is "did the author decide this?", not "is this about performance?".

**Notes are printed, not hidden.** They stay in terminal output, in JSON and in SARIF, and
are excluded only from the exit-code gate. Hiding them would trade a noise problem we have
now measured — 207 of 318 findings, all in one rule — for a trust problem we would not see:
a first run that prints less than it found teaches people the tool is shallow, and that
impression is much harder to reverse than a long list. Anyone who wants them gone has
`ignore = ["TG004"]`, which is the right way to express "I have decided this".

**Per-rule severity overrides only; no per-rule `fail_on`.** They are equivalent in effect —
re-levelling a rule moves it across whatever threshold is set — and one mechanism is easier
to explain than two that interact. `severity` is the existing one, so it wins.

## 8. Open questions

- Should the seven-repo scan be re-run and published as a write-up now that the numbers have
  moved this much? The findings-per-file figure is no longer comparable to the one in the
  README, and the honest version of that post is more interesting than the original would
  have been.
- `note` currently has no representation in the GitHub annotation format beyond `::notice`.
  Worth confirming that a wall of `::notice` annotations on a PR is not its own noise
  problem, since inline annotations are far more intrusive than terminal output.
