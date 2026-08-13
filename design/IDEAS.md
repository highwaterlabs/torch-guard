# Ideas

Parking lot. Nothing here is committed to. Promote to [TODO.md](TODO.md) when we decide to
do it, or to an [RFC](rfcs/) when it needs a design first.

---

## Rules

Each is one file plus a `@register` decorator — the registry was built for this.

- **TG006** `nn.BCELoss` fed raw logits — should be `BCEWithLogitsLoss`, which is also
  numerically stable. Same machinery as TG005, cheap to add.
- **TG007** CPU↔GPU thrashing: `.cpu()`, `.numpy()`, `.item()` inside the training loop.
  Nuance — `.item()` is the *fix* for TG001 but a sync point in a hot loop. The rule has to
  distinguish "once per step" from "once per element", or it will contradict TG001.
- **TG008** non-reproducible runs: `torch.rand` without seeding all of `torch`, `numpy`,
  `random`. Also `DataLoader(shuffle=True)` with no `generator=`.
- **TG009** in-place ops on tensors needed for backward (`x += 1` vs `x = x + 1`). The one
  rule here that genuinely needs alias analysis — probably its own RFC.
- **TG011** `model.eval()` without a matching `model.train()` on the next epoch — a silent
  accuracy killer with batchnorm/dropout.
- **TG012** DDP without `DistributedSampler`, so every rank trains on identical data.
- **TG013** `.to(device)` inside the inner loop instead of hoisted.
- **TG014** loss not divided by accumulation steps under gradient accumulation.

## Analysis engine

- Cross-file resolution: today the provenance analysis stops at file boundaries. A model
  defined in `models.py` and trained in `train.py` is the common layout, and we currently
  lean on naming conventions to bridge it.
- Flow sensitivity within a scope. Would remove the `is_explicitly_detached` discard hack in
  `provenance.py` and let us handle reassignment properly.
- Confidence levels on diagnostics, so heuristic findings can be reported more quietly than
  structural ones.
- Type inference from annotations when present — `def f(x: torch.Tensor)` is free signal we
  currently ignore.

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
