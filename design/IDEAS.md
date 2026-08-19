# Ideas

Parking lot. Nothing here is committed to. Promote to [TODO.md](TODO.md) when we decide to
do it, or to an [RFC](rfcs/) when it needs a design first.

---

## Rules

Each is one file plus a `@register` decorator — the registry was built for this.

- **TG009** in-place ops on tensors needed for backward (`x += 1` vs `x = x + 1`).
  **Decided against.** PyTorch already raises a precise error naming the offending tensor
  and version counter, so a pre-flight check adds nothing the interpreter will not tell you
  a second later — and unlike TG001 or TG011, the failure is loud rather than silent.
  Detecting it properly also needs real alias analysis, which nothing else in the engine
  requires. If it is ever wanted, it deserves its own RFC rather than a rule file.

- **TG015** `pin_memory=True` with `num_workers=0`, which allocates page-locked staging
  buffers that nothing overlaps with. TG004 covers the inverse; this is the wasted half.
- **TG016** `torch.compile` inside the training loop, which recompiles every call and can
  cost more than it saves. Needs care: a guard-triggered recompile is legitimate.
- **TG017** an optimizer constructed inside the training loop, which throws away momentum
  and Adam state every step. Silent, and the loss curve looks like a bad learning rate.

## Analysis engine

- **Memoise per-node derived facts in the dispatcher.** Per-rule cost currently scales
  linearly: on the same 158 files, one rule takes 5.4s and ten take 26.4s, and a full scan
  of torch went from 51s at six rules to 4m18s at thirteen. The single traversal did what it
  promised, but every rule independently recomputes `dotted_name(node.func)` and
  `final_attr(node.func)` on the same nodes. Caching those on the dispatcher, keyed by node
  identity, should flatten most of the per-rule cost without touching any rule.
- **Run the documented pre-commit config, the way `action-smoke.yml` runs the documented
  Action.** The `rev:` pins in the README and `docs/ci.md` had drifted to `v0.1.0` and `v0.2.0`
  and nobody noticed, which is the third instance of one pattern: instructions we publish and
  never execute. The first was `uses: highwaterlabs/torch-preflight@v0` resolving to nothing;
  the second was `action.yml` missing the Node 20 pin bump that every workflow file got. A job
  that installs pre-commit, runs the hook exactly as documented against a known-bad file and
  asserts a non-zero exit would close the whole class rather than the instance. The pins would
  still need bumping at release, but a stale one would fail loudly instead of silently.

- **A scoped-fact helper.** Five separate rules have now hit a bug where a file-level fact
  leaked across functions — `prov.models`, `prov.criteria`, `uses_distributed`, TG008's "does
  this file train", and TG001's deferred-backward exemption. Each was fixed the same way:
  walk out from the current scope, stop at the first binding. That deserves to be one helper
  on `Rule` rather than a pattern everyone re-implements and half of us get wrong the first
  time. TG001 is the case that shows the helper needs *two* modes: `self.*` attributes are
  instance state and should match file-wide, since being written in one method and read in
  another is the normal shape, while bare locals must not.
- Cross-file resolution: today the provenance analysis stops at file boundaries. A model
  defined in `models.py` and trained in `train.py` is the common layout, and we currently
  lean on naming conventions to bridge it.
- **Element types for containers.** The one `pytorch/tutorials` finding that within-file
  return resolution does not clear: `chatbot_tutorial.py`'s `train()` returns
  `sum(print_losses) / n_totals`, and the detach is a hop deeper, inside the elements
  (`print_losses.append(mask_loss.item() * nTotal)`). Resolving it means knowing what a list
  holds, not just what an expression evaluates to. Worth doing only if the shape shows up
  again — one finding is not yet a pattern.
- ~~**Within-file callee return types.**~~ Done. Five of the six `pytorch/tutorials`
  findings; the sixth needs container element types, above. Resolves bare-name calls to
  functions with a single unambiguous return, mapping tuple returns element-wise against
  tuple targets, and only when the return contains a *visible* detaching call — evidence,
  not absence of it. Runs before the fixpoint, since discarding a name afterwards does not
  retract what it already fed.
- **Flow sensitivity within a scope.** Would remove the `is_explicitly_detached` discard
  hack in `provenance.py` and let us handle reassignment properly. The concrete case:
  `transformers/examples/pytorch/language-modeling/run_clm_no_trainer.py` assigns
  `outputs = model(**batch)` twice inside `main()` — line 637 in the training loop, line 665
  inside `with torch.no_grad():`. Same scope, same name, so the flow-insensitive analysis
  merges them and the eval-loop value inherits grad-ness from the training loop. Teaching
  provenance that a `no_grad` assignment produces a detached value was necessary but not
  sufficient; this is what stands between us and the standard Hugging Face eval loop.
  Note the two candidate designs differ in cost: positions on every assignment would pay
  `PositionProvider` on every file, which the dispatcher deliberately avoids, whereas
  extending `ScopePath` to include enclosing statement blocks would not.
- Confidence levels on diagnostics, so heuristic findings can be reported more quietly than
  structural ones.
- Type inference from annotations when present — `def f(x: torch.Tensor)` is free signal we
  currently ignore.

## VRAM model

- **Prefill versus decode peak.** Generation has two phases with different shapes: prefill
  runs the whole prompt at once and does materialise an attention matrix, decode runs one
  token against the cache. We model decode, which is the steady state and the one KV-cache
  sizing is about. The true peak is `max(prefill, decode + cache)`, and with flash attention
  the two converge — without it, prefill can dominate.
- **Paged attention.** vLLM and TensorRT-LLM manage the KV cache in fixed blocks with their
  own allocator, so our contiguous figure is the right order of magnitude but not their
  occupancy. Modelling block granularity and fragmentation would make the estimate usable
  for capacity planning on those runtimes.
- **A "what fits" solver for serving.** The remediation solver answers "how do I make this
  fit". The serving question is the inverse: given a GPU and a model, what is the largest
  batch and context that fit? Same cost model, run backwards.
- **Tensor parallelism for inference.** Sharding is modelled for training (ZeRO/FSDP), but a
  served model split across GPUs divides weights and cache differently.

## Distribution

- Jupyter/`.ipynb` support. Notebooks are where these bugs actually get written.
- LSP server for inline editor diagnostics.
- Flake8/TorchFix plugin shim, per the original doc's compatibility-layer idea — lets teams
  adopt without changing their lint runner.
- `torch-preflight init` to scaffold config + pre-commit + workflow in one command.

## Commercial (post-OSS)

Deliberately parked. The open-source tool has to earn adoption first.

- GitHub App posting inline PR review comments with one-click suggested fixes.
- Cost framing: translate projected VRAM waste into dollars at current A100/H100 rates.
  "This PR wastes $340/month" lands differently than "suboptimal DataLoader config".
- Team dashboard — which repos are at risk, trend over time.
- Custom rule DSL (YAML) so leads can encode internal standards without writing CST visitors.
- Historical calibration: feed real OOM outcomes back to improve the cost model. This is the
  data moat — nobody else would have measured OOM outcomes tied to source configs.

## Landscape (checked 2026-08-13)

**TorchFix is archived.** `meta-pytorch/torchfix` (formerly `pytorch-labs/torchfix`) —
152 stars, 18 forks, 20 open issues, last push 2025-08-23, repo marked archived. It still
pulls **~55,000 PyPI downloads a month**, so the usage is real and the maintenance is not.

Read it both ways. There is a vacancy in exactly our niche with demonstrated demand — and
also a caution, since an official-adjacent Meta project here did not sustain. The likeliest
explanation is scope rather than absence of need: TorchFix was a flake8 plugin for
deprecation warnings, which is a much narrower proposition than catching OOMs and silent
convergence bugs. Worth revisiting that assumption rather than assuming the gap is free.

Our public docs do not mention TorchFix, so nothing is stale. If comparison material is
ever written, note it is archived rather than implying it is a live competitor.

**`torchguard` (no hyphen) blocked the name `torch-guard` on PyPI.** It has 38 downloads
a month and every project URL points at `github.com/yourusername/torchguard`, which 404s —
the packaging template placeholders were never filled in. It solves a different problem
(per-sample error tracking for `torch.compile` regions).

None of which matters, because PyPI rejects a name that is identical after separators are
stripped: `torch-guard` normalises to `torchguard`. A 404 from `GET /pypi/torch-guard/json`
means only that no project holds that exact string — it does **not** mean the name can be
registered. Checking exact availability and calling it free is how we ended up choosing a
name we could not publish. The project is now `torch-preflight`.

PEP 541 was not an option: `torchguard` shipped a release in January 2026, so it does not
meet the abandonment bar however dead the repo link is.

## Open questions worth thinking about

- Should `check` and `estimate` be one command? Running the estimator on every PR is the
  point, but it is meaningfully slower and can need network.
- How do we handle repos with many training scripts — estimate all of them, or only the ones
  reachable from a declared entry point?
- Is there a story for JAX/Flax, or does that dilute a tool whose whole value is depth in
  one framework? Current instinct: dilutes it.
