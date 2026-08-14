# TODO

Decided work, grouped by phase. Delete items when done — git remembers them.
Undecided things live in [IDEAS.md](IDEAS.md).

---

## Phase 0 — static linter core ✅ done

TG001–TG005, provenance analysis, engine, autofixer, 4 reporters, CLI, config,
suppressions, pre-commit hooks, GitHub Action, CI matrix.

## Phase 1 — VRAM static tier ✅ done

Per RFC [0001](rfcs/0001-vram-estimator.md). No new **required** dependencies.

- [x] `vram/hardware.py` — 24 GPUs + 21 cloud instances, usable (not advertised) capacity
- [x] `vram/costmodel.py` — six-term model, constants isolated in a CALIBRATION block
- [x] `vram/archdb.py` + bundled snapshot — 40 architectures, ships in the wheel
- [x] `vram/extract.py` — RunConfig from the CST, with per-field provenance
- [x] `vram/solver.py` — remediation search with mutual-exclusion groups and a capped
      greedy stack
- [x] `torch-preflight estimate` / `torch-preflight gpus`
- [x] Risk banding with error intervals; no fabricated probability
- [x] TG010, gated on `target_gpu`, silent at low confidence, never touches the network
- [x] `tests/calibration/` — parameter formula enforced to 3% against published counts
- [x] Base-install CI job asserting torch/huggingface_hub never load

### Calibration ✅ activation side done

- [x] Spike [0001](spikes/0001-meta-device-activation-capture.md) — GO. `saved_tensors_hooks`
      works on meta, dedup via `storage._cdata`, fused attention is *not* a blind spot
- [x] `tests/calibration/measure_activations.py` — measures the activation coefficients on
      the meta device, no GPU required
- [x] Coefficients are now architecture-dependent and measured: dropout triples the
      quadratic term, so Llama-class models are no longer over-charged (−44% on the
      llama-2-7b activation estimate)
- [x] Constants pinned by tests — changing one without re-measuring fails the suite

### Before shipping

- [x] **Triaged every torch finding.** My guess that they were "plausibly intentional
      retention" was wrong: four of the five were our bugs. 10 findings -> 5, all of the
      survivors deliberate graph retention. Each fix is regression-tested:
      - **`models_to_test` read as a validation dataloader.** In a test suite half the
        identifiers contain "test". An eval-loader name must now also look like something
        you iterate batches from (`loader`/`dataset`/`batches`/...).
      - **`get_loss(...)` treated as a torch functional** because it ends in `_loss`,
        returning grad-bearing regardless of its arguments. That suffix heuristic now
        applies only to the known `F.*` functionals when called by a bare name.
      - **`torch.autograd.grad(...)`** returns detached tensors unless `create_graph=True`.
      - **Name leakage across functions.** `prov.models` was a flat, file-wide set, so
        `prepared = DistributedDataParallel(...)` in one helper made an unrelated
        `prepared` a thousand lines away look like a model. Both `models` and `grad`
        lookups are now scope-aware and stop at the first scope that binds the name — the
        same bug also let an outer `loss` leak into a nested helper with its own `loss`.

      Remaining 5, all judged true-but-intentional (a `# noqa` case, not a rule change):
      - `_inductor/fx_passes/numeric_utils.py:151,152` — TG003, a gradient-comparison
        harness that deliberately calls `backward(retain_graph=True)` in an optimizer loop
      - `distributed/pipelining/schedules.py:304` — TG001, pipeline parallelism must hold
        microbatch losses until their backward runs
      - `dist_autograd_test.py:2086,2098` — TG001, a dict deliberately chaining
        graph-attached tensors across ranks

      Reproduce with: `torch-preflight check <site-packages>/torch -f json`

### Known gaps that came out of building it

- [x] **CNN activation memory measured** for all eleven vision models, on the meta device
      with no GPU — I had wrongly filed this as needing one. Batch-linearity and
      area-scaling are verified at measurement time, not assumed.
- [x] `CUDA_CONTEXT_BYTES` (135 MiB) and `FRAGMENTATION_FRACTION` (0.105) measured on a
      Tesla T4; end-to-end peaks recorded for GPT-2, BERT, DistilBERT and ResNet-50. Mean
      absolute error against the eight measured peaks is **3.7%**.
- [x] **The meta-measured CNN activations check out against a real allocator.** ResNet-50
      on a T4 came in at +5.6% (batch 16) and +1.4% (batch 32) — the first end-to-end
      confirmation that measuring on a device that allocates nothing predicts a device that
      does. A re-measurement with the vision runs included puts fragmentation at 0.098
      against the shipped 0.105; the difference is within run-to-run spread and the shipped
      value errs high, so it stands.
- [x] **LM-head retained bytes measured**: exactly 4.00 per logit element, across five
      shapes and three vocabularies, precision-independent. Split out from the fitted
      backward-transient part so the evidence for each is visible.
- [x] **LM-head backward transient measured**, replacing the two-point fit. The sweep it
      needed now exists (`measure_cuda.py --lm-head-sweep`): four vocabularies from 8k to
      128k at two batch sizes on a tiny body, eight peaks spanning 16x in logit count.
      Least squares gives 15.72 bytes per logit of peak, 14.22 after dividing out
      fragmentation, minus the measured 4 retained — so the constant went 6 -> 10 and mean
      absolute error 4.4% -> 3.7%. Still *not* the fixture optimum of 14: GPT-2 is the only
      measured peak with an LM head, so that "fit" is two points again, and they disagree
      in sign.
- [ ] **The per-logit cost is not batch-invariant** — ~19.7 bytes/logit at batch 4 against
      ~14.7 at batch 8, consistently across all four vocabularies. A single constant cannot
      express that, which is why GPT-2 at batch 8 stays ~12% under. Worth understanding
      before adding a batch term: it is more likely an allocator reuse effect at peak than
      a real difference in bytes required.
- [x] **`VRAMGuard` ignored activations entirely** — `profile_live_model` returns exact
      parameter counts and no shape, so the term silently read zero. Measured against a
      real ResNet-50 at batch 32 the guard projected 0.61 GiB where the card peaked at
      1.86: **−67.5%**, and *under*-estimating is the direction that keeps a guard quiet
      through the OOM it exists to prevent. It now measures activations from the live
      module via `functional_call` on meta parameters — no allocation, no mutation of the
      caller's model — giving +8.8% on the same case.
- [ ] **The guard's measurement is forward-only**, so for a language model it sees the
      logits but not the loss temporaries that peak during backward. The static estimator
      models those; the guard does not.
- [ ] **Calibration covers one GPU.** `CUDA_CONTEXT_BYTES` is a single T4 data point;
      `hardware.Gpu.context_mib` holds per-card overrides as more arrive. An A100/H100
      run would be the most valuable next measurement.
- [x] **Encoder-decoder activations measured** for T5 and Whisper (8 snapshot entries),
      so they estimate instead of reporting unknown. A decoder-only formula cannot express
      these: there are two sequence lengths, and a decoder layer carries a third attention
      block whose K/V projections run at the *encoder* length.
      `tests/calibration/measure_encoder_decoder.py` fits per-family coefficients on the
      meta device. Validated against direct measurement to **2.5% worst case** over 12
      shapes, including whisper-medium and t5-large which were held out of the fit.
      `params_from_transformer_shape` now counts cross-attention too (T5 exact, Whisper
      within 2.7% -- its conv frontend and learned position tables are not in the formula).

      Three separate collinearities had to be broken, and **every degenerate version
      reported a better residual than the correct one**, which is worth remembering the
      next time a fit here looks clean:
      - encoder-linear against cross-KV: identical columns when `L_enc == L_dec`, which is
        true of every T5 and Whisper size. 0.00% residual, enc_linear 26.84 against a true
        48.34. Fixed by measuring the encoder alone first.
      - linear against quadratic, and decoder-linear against cross-attention: Whisper's
        encoder length is fixed at 1500 and every size uses head_dim 64, so those columns
        are exactly proportional -- unidentifiable in principle. Unconstrained it returned
        `dec_linear = 0.16`. Both quadratic terms are pinned to the separately measured
        attention coefficient; T5, where the split *is* identifiable, fits cross-attention
        free at 6.03 against the pinned 6.0.
- [x] **Grouped-query attention: the premise was wrong, no change needed.** I had filed
      this as "the activation formula ignores kv_heads and will over-estimate GQA models".
      Measured, it does not. `transformers.repeat_kv` expands K/V to the full head count
      and reshapes; reshaping a non-contiguous expand copies, so autograd retains full-size
      K/V exactly as MHA would — retained bytes are bit-identical across kv_heads 16/8/4/2.
      GQA saves parameters (already applied) and KV cache (not modelled, inference-only),
      not training activations. `enable_gqa=True` does avoid it (0.65x at an 8x ratio), but
      in transformers 5.x that path exists only in the exporters and one model. Charging
      the cheap rate would under-estimate every mainstream GQA model. Both directions are
      now pinned by tests so this cannot be "optimised" back.

      Note for anyone re-measuring: this must be done on the **CPU**, not the meta device.
      `enable_gqa` falls back to the math backend on meta and materialises the expanded
      K/V, so every variant measures identical and the effect is invisible. Spike 0001's
      "fused attention is not a blind spot" holds for plain SDPA but not for this flag.
- [ ] **KV cache is not modelled at all.** Irrelevant for training, but it dominates
      inference memory and is the place GQA genuinely pays off (8x on Llama-3-70B).
      `--inference-only` currently just scales the activation term.
- [x] **DeepSpeed ZeRO stage is now read, not assumed.** The comment said the stage "is in
      a JSON config we cannot read", which was untrue: it is either a dict literal in the
      same file or a path to a JSON file beside it, and reading JSON is not executing code.
      Handles `deepspeed.initialize(config=...)` by path, by variable and inline, plus
      `TrainingArguments(deepspeed=...)`. Falls back to the old stage-2 assumption when the
      config is genuinely unresolvable (built by a function call). Path traversal outside
      the source tree is refused, and missing or malformed JSON degrades quietly.
      This matters: stage 3 shards parameters too, so it is the difference between a 70B
      model fitting and not.
- [ ] DeepSpeed **offload** (`offload_optimizer` / `offload_param`) moves state to CPU and
      is now visible in the configs we parse, but is not modelled — we still charge it to
      the GPU, which over-estimates offloaded runs.
- [x] **GitHub Actions pins bumped off Node 20**: checkout v4->v7, setup-python v5->v7,
      upload-artifact v4->v7, download-artifact v4->v8. PR CI exercises `ci.yml`, but
      `release.yml` only runs on a tag — rehearse it via `workflow_dispatch` (which builds
      and checks without publishing) before the next release, since the artifact
      upload/download pair is the part CI does not cover.

## Phase 2 — exact tier

- [x] Run spike [0001](spikes/0001-meta-device-activation-capture.md) — **GO**
- [x] `vram/providers/meta.py` + `[vram]` extra. Exact parameter counts and measured
      activations for arbitrary models via `module:factory` entry points, with
      `--model-args key=value`. Both spike traps are handled: parameter storages are
      excluded and dedup keys on `storage._cdata`. Cross-checked against
      `params_from_transformer_shape` — measuring and deriving agree, which is the
      strongest validation available without a GPU. Failures degrade to UNKNOWN rather
      than crashing, and `check` never reaches this path.
- [x] Model autodetection Layer 2 — `vram/autodetect.py`. Folds constructor arguments and
      module-level constants, resolves the class through the file's own imports, and
      meta-instantiates it. Guarded by an import-safety check: a module whose top level
      builds objects or calls functions is refused, because importing it would do that
      work for real. Unresolvable arguments are named, never guessed.
- [x] `[hub]` integration tested properly. Real `config.json` files are captured in
      `tests/fixtures/hub/` and replayed offline; a `network`-marked test (deselected by
      default, `pytest -m network`, plus a non-blocking CI job) hits the live hub.
      **This found two bugs the hand-written offline tests could not**, because I had
      authored both the field mapping and the fixtures so they agreed:
      - `tie_word_embeddings` absent means **True** in transformers, not False. Defaulting
        to False double-counted `vocab x hidden` and put GPT-2 31% over its real size.
      - DistilBERT names its fields `dim` / `hidden_dim` / `n_heads` / `tie_weights_`, and
        `hidden_dim` is the FFN width, not the model width. It resolved to nothing at all.
      All four captured models now derive within 0.1% of their published counts.

## Phase 3 — runtime ✅ done

- [x] `VRAMGuard` context manager — exact parameter/gradient/optimizer accounting from the
      live model, inferring optimizer kind and precision from the objects themselves.
      Raises only on `CERTAIN_OOM`; anything less certain warns, because aborting a job on
      a guess is worse than the OOM it would prevent. `strict=True` opts into raising.
      Exported lazily from `torch_preflight` so the base install still never imports torch.
- [x] Verification against `torch.cuda.max_memory_allocated()` on exit, exposed as
      `guard.measured_peak` and `guard.accuracy`.
- [ ] Feed real `guard.accuracy` measurements back into the calibration fixtures — needs a
      GPU session, same as the remaining CUDA constants.

### Rules beyond the first five

- [x] **TG006 — binary cross-entropy activation mismatch.** Four cases: `sigmoid` into
      `BCEWithLogitsLoss` (double sigmoid, error, autofixable when inline), the same via a
      variable, raw logits into `BCELoss` (`nan` on the first negative value, error), and
      `sigmoid` + `BCELoss`, which is *correct* but numerically fragile, so it warns rather
      than errors. Clean on torch's 2,285 files after one round of triage.

      Two bugs found while building it, both now regression-tested:
      - **`Provenance.criteria` was a flat name -> class map**, so two functions each
        binding `crit` collided and whichever was parsed last decided the class for both.
        A correct `BCELoss` call was reported as a double-sigmoid error against
        `BCEWithLogitsLoss`. Now scope-aware, matching `grad` and `models`. **This
        affected TG005 too.**
      - **Three false positives in `torch/testing/_internal/common_nn.py`**: flagging any
        `nn.Sigmoid()` construction in a file that mentioned `BCEWithLogitsLoss` anywhere.
        A bare `sigmoid = nn.Sigmoid()` local used to build a reference implementation is
        not a model ending in a sigmoid; only final position in an `nn.Sequential` is.
- [ ] TG007-TG009 and TG011-TG014, listed in [IDEAS.md](IDEAS.md). TG007 (CPU-GPU
      thrashing) needs care: `.item()` is the *fix* for TG001 but a sync point in a hot
      loop, so the rule has to tell "once per step" from "once per element" or it will
      contradict a rule we already ship.

## Cross-cutting

- [x] Name and org settled: package `torch-preflight`, org `highwaterlabs`, deliberately
      distinct so the company is not named after another project's trademark.
      `torch-guard` had to be abandoned: PyPI rejects names that collide after separators
      are stripped, and it normalises to the existing `torchguard`. Exact-name
      availability is not registerability — see IDEAS.md.
- [x] Public repo live at `highwaterlabs/torch-preflight`, MIT, CI green. The private
      `torch-preflight-cloud` repo is still not needed; RFC 0002 lives there when it exists.
- [x] "What stays free" section in the README, per RFC 0002 §6.
- [x] `design/` is tracked and public; README links verified.
- [x] Committed and pushed; README split into `docs/` and rewritten (PRs #1, #2).

- [x] **0.1.0 published to PyPI**, verified by installing from PyPI into a clean
      virtualenv: 9 packages, no torch, CLI and estimator both working.
      The first release attempt failed because the rename lived only on a local branch,
      so `main` — and the tag — still packaged `torch-guard`. The build validated the
      version against the tag but never the name, so it reached the upload step and died
      on an opaque OIDC rejection. A name guard now fails at build time instead.
- [x] **0.2.0 published**, the VRAMGuard activation fix. Minor rather than patch: the guard
      now fails and warns on configurations it previously waved through. Every release gate
      was run locally before tagging, because a PyPI version number can never be reused.
      Releases exist for both versions; repo description, topics and logo are set.
- [x] *Declined:* renaming the local working directory from `torch-guard` to
      `torch-preflight`. Kept as-is deliberately.
- [ ] **Version the private repo.** RFC 0002 sits unversioned in `~/Dev/torch-preflight-cloud/`.
- [x] Replaced the hardcoded test-count badge with live PyPI version, Python-version and
      licence badges, which update themselves.
- [ ] **Delete the merged branches from the remote.** Eight are fully merged into `main`
      and still there: `calibration/cnn-and-lm-head`, `docs/colab-download-fix`,
      `fix/guard-activation-measurement`, `feat/tg006-bce-logits`,
      `chore/zero-stage-and-snapshot`, `chore/todo-reconcile`, `feat/gqa-activations`
      and `feat/encoder-decoder-shapes`. An earlier entry claimed this was already done
      "locally and on the remote", which was only ever true locally.
- [x] **Snapshot refresh process defined**: `tests/calibration/verify_snapshot.py`
      compares the bundled snapshot against the live hub configs. Deliberately a verifier
      rather than a regenerator, so a renamed upstream field cannot silently rewrite the
      numbers in a diff too large to review. Cadence and rationale in
      `tests/calibration/README.md`. Currently 6 entries verified clean.
- [x] **Stress-tested against torch's own source** (2285 files). False-positive rate after
      fixes: **0.0033 findings/file** — 3 findings in 900 files, all "true but intentional"
      deliberate graph retention (pipeline parallelism, distributed autograd tests). Three
      real rule bugs found and fixed, each now regression-tested:
      - `x.requires_grad = <expr>` seeded grad on any value, not just `True`; a detached
        leaf now also clears any seed (hit `torch.utils.checkpoint`)
      - "loss" matched mid-word in helper names with no grad flowing in
        (`_multilabelmarginloss_reference`); loss-named helpers now need a grad-bearing arg
      - `dist_autograd.backward(ctx, [loss])` accumulates into an RPC context, not `.grad`
      - TG003 now also requires an optimizer step in the loop: `.backward()` with nothing
        applying the gradients has nothing to go stale

- [x] **PERFORMANCE fixed: 14 min -> 51 s on torch (2285 files), a ~16x speedup.**
      Three changes, each measured:
      1. **One shared traversal** for all rules (`RuleDispatcher`) instead of one walk per
         rule — rules are no longer visitors, they are handlers reading dispatcher-owned
         state. 367 -> 92 ms/file (4.0x).
      2. **Deferred position resolution.** `PositionProvider.resolve` cost 3.1 s on a
         1.3 MB file — *more than parsing it* — and ~99.7% of files have no findings at
         all. Diagnostics now carry the node and positions are filled in afterwards, only
         when a file actually reported something. 92 -> 67 ms/file (5.5x cumulative).
      3. **Process pool across files**, with `--jobs`. Sequential when `--fix` is on,
         because the fixer replaces nodes by identity in the tree the rules ran against.
         Falls back to sequential if the environment refuses to fork.
      Parallel and sequential results are asserted identical in `tests/test_engine.py`.
