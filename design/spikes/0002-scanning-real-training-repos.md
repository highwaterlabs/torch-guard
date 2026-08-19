# Spike 0002 — What happens when you point the tool at real training code

**Question:** we had been scanning PyTorch's own source. If we scan repositories whose
*product* is training runs, do we find bugs worth reporting upstream?

**Answer:** one. And sixteen in ourselves.

**Dates:** 2026-08-17 to 2026-08-19
**Status:** complete; the fixes shipped across #43, #46, #49, #50, #51

---

## 1. Why

The idea was to get traction by filing PRs against well-known repositories. Before writing
any of them, the findings had to be read.

The first thing that read-through changed was the target. torch-preflight had been validated
against `torch` itself, which is the wrong shape of codebase for it:

| | files | findings | per file |
|---|---|---|---|
| `torch` (a framework) | 2,285 | 20 | 0.0087 |
| seven training repos | 1,615 | 318 | 0.20 |

A 23× difference, and it is structural rather than luck. Our rules are about *training
scripts* — retained graphs, gradient accumulation, `DataLoader` settings, seeding. A framework
contains almost no training loops, so 18 of torch's 20 findings sit in
`torch/testing/_internal/` where a deliberate `cuda.synchronize()` in a distributed test costs
nothing. Zero of them are worth a PR.

So: `pytorch/examples`, `pytorch/tutorials`, `torchtune`, `litgpt`, `trl`, `LLaMA-Factory`,
and `transformers/examples/pytorch`. A bug in an *examples* repo is also worth more than the
same bug in core, because those files get copy-pasted into real projects.

## 2. Method

Shallow clones, `torch-preflight check <repo> --format json`, then **every finding read by
hand against its source** — not clustered by message, and at every severity.

Reading the low-severity ones mattered more than expected; see §5.

## 3. What we found in them

**One PR.** `pytorch/examples/distributed/tensor_parallelism/` — all three examples call
`backward()` then `optimizer.step()` in a loop with no `zero_grad()` anywhere in the file, so
every iteration steps on the running sum of all previous gradients. Toy loops, so nothing looks
broken, but these files are a common starting point. Three one-line additions, placed after
`step()` to match the sibling `distributed/FSDP2/example.py`. Filed as
[pytorch/examples#1424](https://github.com/pytorch/examples/pull/1424).

A second finding in the same directory was deliberately left out: `sequence_parallel_example.py`
has no `manual_seed` where its two siblings do, but its own comment says the input "can be
different across all ranks", so it reads as intentional. Bundling it would have given a reviewer
something to argue about.

**8 true findings we chose not to file.** torchtune's recipes do `running_loss += current_loss`
without `.detach()` — and their own `full_finetune_single_device.py` detaches, so the idiom is
theirs and the others are oversights. Real, but small, and not our place to churn eleven files
over.

## 4. What we found in ourselves

Sixteen distinct causes. Grouped by the kind of mistake rather than by rule, because the kinds
repeat:

### A name was trusted over the code right next to it

- **`self.softmax = nn.LogSoftmax(dim=1)`** feeding `NLLLoss` is correct, and is what PyTorch's
  own char-RNN tutorial does. TG005 read the *attribute name* and reported a convergence bug,
  with the constructor two lines above.
- **`self.softmax = nn.Softmax(dim=1)` in a GAT layer** normalises attention coefficients; the
  model correctly ends in `F.log_softmax`. Constructing the layer was treated as evidence of the
  model's output activation. Attention softmax is in every transformer and GNN.
- **`self._grad_scaler = training.scale_grads_`**, then called through the alias so it could be
  wrapped in `torch.compile`. TG014 matched on the call-site name and saw nothing.
- **`AutoTokenizer.from_pretrained(...)`** was a model, because `from_pretrained` is in
  `MODEL_WRAPPERS`. So `tokenizer(x["question"])` was a forward pass and TG002 reported a
  missing `no_grad` around *tokenisation*.
- **`train()` returning `loss.item() / n`** — the caller's `total_loss += loss` accumulates a
  float, but the caller's variable is named `loss` and the name hint is the strongest heuristic
  in the analysis. The callee's `return` was a few lines below, in the same file.

The pattern: **read the binding, not the name.** Four of these were the same lesson arriving in
different rules, twice in one afternoon.

### The rule did not know what it was about

- **TG007 flagged its own recommended fix.** All six findings were
  `correct += (predicted == labels).sum().item()` in a validation loop — *verbatim* what its
  hint tells you to write. Its batch-loop exemption matched iterable **names** (`loader`,
  `dataloader`, `batches`) and missed `dev_iter` and `valloader`. The fix was not a longer name
  list but requiring evidence of per-element iteration.
- **TG013 treated a download as a redundant upload.** `pinmem_nonblock.py` — a tutorial whose
  subject is measuring transfer behaviour — loops 100 times over `tensor.to("cpu")` on a tensor
  created with `device="cuda"`. Wrong twice over.
- **TG013 told `fast_neural_style` to hoist a device restore.** It moves the model to the host
  to checkpoint and back afterwards; following the advice would leave it on the host.
- **TG002 said `test()` "never calls `.backward()`"** with the call nine lines below. FGSM
  iterates `test_loader` and backwards through it on purpose, because an adversarial attack
  needs gradients w.r.t. the input.

### Retention that a backward pass depends on

TG001's exemption was syntactic, and syntax cannot decide this:

```python
loss = torch.stack(losses).mean()   # a throwaway reduction, or the training loss?
```

It is a leak in Lightning's epoch-end logging and load-bearing in `trl`'s distillation
objective, written identically. Rewritten around reachability. Three separate shapes had to be
handled: an element read by a getter (pipeline-parallel schedules), an accumulator returned for
backward (chunked loss modules), and a container reduced into a value that is returned
(RL objectives). A fourth — a returned **bare name** — needed the extra step that it counts
unless the name is *itself* the container.

### Autograd state we were not tracking

- **Values produced under `no_grad`** propagated a graph, because the check was positional and
  the standard evaluation loop wraps only the forward.
- **`accelerator.backward(loss)`** seeded `accelerator` as a live tensor. Accelerate, Fabric and
  DeepSpeed engines invert the usual shape: the tensor is the argument. Every later
  `accelerator.gather_for_metrics(...)` then read as grad-bearing.
- **Flow insensitivity between sibling loops.** One `main()` binds `outputs` in the training
  loop and again under `no_grad` in the evaluation loop; both shared one key.

### A claim that was measured and found overstated

TG001 reported every graph-attached store as `CRITICAL_OOM`. `backward()` frees each node's
saved tensors as it traverses, so a tensor stored *after* its backward retains none of them.
Measured on a 4-layer transformer ([`measure_retention.py`](../../tests/calibration/measure_retention.py)):

| | activations retained | host memory |
|---|---|---|
| already backwarded | **0 KiB/iter** | ~15 KiB/iter |
| never backwarded | **186 MiB/iter** | ~15 KiB/iter |

186 MiB/iteration OOMs an 80 GB A100 in ~440 steps. 15 KiB/iteration is 1.4 GiB per 100k steps —
real, unbounded, and survivable. Roughly 12,000× apart, in different memory, and the gap widens
with model size because activation cost scales with the model while node cost does not.

**Our own README example was the smaller case.** It now shows both and explains the difference.
The split forced [RFC 0003](../rfcs/0003-severity-and-ci-gating.md), which defines what the
three severities mean, because `warning` had been holding both that leak and "you did not set
`pin_memory`".

## 5. Two things about method, which are the real output

### A wild scan is evidence about false positives only

While fixing the `no_grad` case we introduced a false negative: detachment propagated whenever
a binding was not *provably* grad-bearing, treating absence of proof as evidence. That silenced
a real leak **even when `loss.backward()` was called on the name.**

Neither check caught it:

- **All 437 tests passed**, because every TG001 fixture assigns from something resolvable like
  `criterion(model(batch), y)`. None used a callee the analysis cannot see through.
- **The scan reported zero new findings** — which is structurally blind here. A true positive
  that stops firing is indistinguishable from a false positive that got fixed.

The scan had reported 24 removals. **Fourteen were findings being silenced**, eight of them
genuine. The honest figure was ten.

So: false negatives need fixtures that *deliberately* exceed what the analysis can resolve, and
there are now two of those, including the case that should have been impossible to get wrong.

### Read every severity, not just the errors

TG007 was the worst rule in the set — 6 of 6 false, every one the rule reporting its own advice
— and it had never been re-read once, across three separate passes over the findings, **because
it never produces an error.** Errors get attention; a rule that only ever warns can stay wrong
indefinitely.

## 6. Result

| | first scan | after |
|---|---|---|
| errors | 63 | **11** |
| warnings | 255 | 46 |
| notes | — | 207 |
| total | 318 | 264 |

54 findings removed, no new ones, each removal traced to a named cause and regression-tested
against the file that exposed it. TG005, TG014 and TG007 went to zero.

**TG004 came out accurate.** A sample of the 207 notes were all genuine `DataLoader` calls
missing `num_workers` or `pin_memory`. Its problem was volume, not correctness, which making it
a note under RFC 0003 already solved. And `litgpt` scanned clean throughout — 137 files, zero
errors, before and after — so none of this was "the tool fires everywhere".

## 7. Still open

- ~~**TG008 (31 findings)** should be *split*, not demoted.~~ Done. No seeding at all is a choice we often
  cannot see, since seeding frequently lives in the launcher rather than the script — that is a
  `note`. **Partial** seeding, torch seeded and NumPy not, is a defect whose intent is visible in
  the code, and stays a `warning`. On this corpus that is 30 notes and 1 warning, and the one
  that remains is the informative one.
- **The `label` naming heuristic on non-tensor lists** no longer produces a visible finding.
  Deliberately not fixed: changing a heuristic with no failing case to verify against is exactly
  how the false negative in §5 happened.
- **Container element types.** `chatbot_tutorial.py` returns `sum(print_losses) / n_totals`,
  where the detach is inside the list elements. Needs knowing what a list *holds*. One finding
  is not yet a pattern.

## 8. Would we do it again

Yes, but not for the reason we started. As a source of upstream PRs it returned one three-line
fix from 1,615 files. As a source of *our own* bugs it returned sixteen, several in rules that
had been shipped and released for weeks and looked fine against `torch` and against our own
fixtures.

The generalisable part: our test fixtures are written by the same person with the same
assumptions as the rules, so they agree with the rules by construction. Real code disagrees. The
cheapest way to find out what a rule actually claims is to point it at a corpus nobody wrote
for it — and then read the output, including the parts that cannot fail a build.
