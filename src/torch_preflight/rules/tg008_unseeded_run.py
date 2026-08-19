"""TG008 - a training run whose randomness is not seeded."""

from __future__ import annotations

from typing import Dict, List, Set

import libcst as cst

from ..analysis.helpers import contains_call_to, dotted_name, final_attr
from ..diagnostics import Category, Severity
from .base import Rule, register

#: Random sources, grouped by the generator each one draws from. Seeding one does nothing
#: for the others, which is why partial seeding is the common failure rather than none.
TORCH_RANDOM = {
    "rand", "randn", "randint", "randperm", "rand_like", "randn_like", "randint_like",
    "bernoulli", "multinomial", "normal", "poisson",
}
NUMPY_RANDOM_MODULES = ("np.random", "numpy.random")
STDLIB_RANDOM_CALLS = {
    "random", "randint", "randrange", "choice", "shuffle", "sample", "uniform", "gauss",
}

#: Seeding calls, by generator.
TORCH_SEEDS = {"manual_seed", "manual_seed_all"}
NUMPY_SEEDS = {"seed", "default_rng"}

#: One call that seeds everything: Lightning's `seed_everything`, transformers' `set_seed`,
#: and the convention most projects adopt for their own helper.
SEED_EVERYTHING = {"seed_everything", "set_seed", "set_random_seed", "seed_all"}

_LABELS = {
    "torch": ("`torch.manual_seed(...)`", "torch"),
    "numpy": ("`np.random.seed(...)`", "NumPy"),
    "random": ("`random.seed(...)`", "the `random` module"),
}


@register
class UnseededRun(Rule):
    code = "TG008"
    name = "unseeded-run"
    summary = "Training run with unseeded randomness"
    severity = Severity.WARNING
    category = Category.CONVERGENCE_BUG
    explanation = """
Weight init, dropout masks, shuffling and augmentation all draw from random generators, and
there are **three independent ones** in a typical PyTorch script: torch's, NumPy's and the
standard library's. Seeding one does nothing for the other two.

The consequence is not a crash but an unanswerable question. When a run scores two points
worse than yesterday's, there is no way to tell whether the change caused it or the seed did,
so the experiment cannot be repeated and the result cannot be attributed. Partial seeding is
the usual shape: ``torch.manual_seed(42)`` is set, the augmentation pipeline draws from
``np.random``, and the run still varies between invocations.

Seed every generator you draw from::

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

or use one helper that covers all three — Lightning's ``seed_everything(seed)`` or
transformers' ``set_seed(seed)``.

Two things this rule deliberately does not claim:

* Seeding does not make a CUDA run bit-identical. Non-deterministic kernels still vary; that
  needs ``torch.use_deterministic_algorithms(True)`` and it costs throughput.
* ``DataLoader(shuffle=True)`` with ``num_workers > 0`` seeds each worker from the base seed,
  so it is covered by ``torch.manual_seed`` — no separate ``generator=`` is required for
  reproducibility, only for independence from other torch draws.

It only fires on files that actually train (something calls ``.backward()``). A library that
draws randomly and leaves seeding to its caller is behaving correctly.

The two cases are reported at different levels, because they are different findings:

* **Partial seeding is a warning.** The intent is visible in the code — someone asked for
  reproducibility — and a generator is escaping anyway. That is a defect.
* **No seeding at all is a note.** It is often a choice, and often one this tool cannot see:
  seeding frequently lives in the launcher or the job scheduler rather than the training
  script, and some runs deliberately want variance across invocations.
""".strip()

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._used: Dict[str, cst.CSTNode] = {}
        self._seeded: Set[str] = set()
        self._seeds_everything = False
        self._trains = False

    # ---------------------------------------------------------------- collection

    def visit_Call(self, node: cst.Call) -> bool:
        dotted = dotted_name(node.func) or ""
        leaf = final_attr(node.func) or ""

        if isinstance(node.func, cst.Attribute) and leaf == "backward":
            self._trains = True

        if leaf in SEED_EVERYTHING:
            self._seeds_everything = True
            return True

        # -- seeding
        if leaf in TORCH_SEEDS and ("torch" in dotted or dotted == leaf):
            self._seeded.add("torch")
        if leaf in NUMPY_SEEDS and any(m in dotted for m in NUMPY_RANDOM_MODULES):
            self._seeded.add("numpy")
        if leaf == "seed" and dotted in ("random.seed",):
            self._seeded.add("random")
        if leaf == "manual_seed" and "cuda" in dotted:
            self._seeded.add("torch")

        # -- drawing
        #
        # An explicit `generator=` is a deliberately controlled source, seeded through its
        # own `Generator`. torch's own dist_optimizer_test does exactly this to avoid
        # cross-rank divergence, and counting it as unseeded was wrong.
        if any(arg.keyword is not None and arg.keyword.value == "generator"
               for arg in node.args):
            return True

        # The draw must belong to code that trains. A random helper in a file that happens
        # to contain a `.backward()` somewhere else is a library function, and seeding is
        # its caller's business -- this is the same file-wide leakage that has now bitten
        # `models`, `criteria`, `uses_distributed` and this rule.
        if not self._draw_is_in_training_code():
            return True

        if any(m in dotted for m in NUMPY_RANDOM_MODULES) and leaf not in NUMPY_SEEDS:
            self._used.setdefault("numpy", node)
        elif dotted.startswith("random.") and leaf in STDLIB_RANDOM_CALLS:
            self._used.setdefault("random", node)
        elif leaf in TORCH_RANDOM and (dotted.startswith("torch.") or dotted == leaf):
            self._used.setdefault("torch", node)
        return True

    def _draw_is_in_training_code(self) -> bool:
        """Module level, or inside a function that itself calls ``.backward()``."""
        function = self.current_function
        if function is None or function.node is None:
            return True  # module level: it governs whatever the script does
        return contains_call_to(function.node, ["backward"])

    # ------------------------------------------------------------------ verdict

    def leave_Module(self, original_node: cst.Module) -> None:
        if self.ctx.is_lightning or self._seeds_everything:
            return
        # Only training scripts own their seeding; a library leaves it to the caller.
        if not self._trains or not self._used:
            return

        missing: List[str] = [
            source for source in ("torch", "numpy", "random")
            if source in self._used and source not in self._seeded
        ]
        if not missing:
            return

        node = self._used[missing[0]]
        calls = ", ".join(_LABELS[source][0] for source in missing)
        drawn = " and ".join(_LABELS[source][1] for source in missing)

        # Two findings, not one, and RFC 0003's test ("is this code defective, or merely
        # untuned?") gives them different answers.
        #
        # Partial seeding is a *defect*: the author's intent is visible in the code -- they
        # asked for reproducibility -- and a generator is escaping anyway. That is a warning.
        #
        # No seeding at all is a *choice*, and often one we cannot see. Seeding frequently
        # lives in the launcher or the scheduler rather than the script, and this tool only
        # reads the script; some runs deliberately want variance across invocations. That is
        # a note.
        #
        # On seven real training repos the split is 30 notes and 1 warning, and the one that
        # stays a warning is the informative one.
        if self._seeded:
            already = " and ".join(sorted(self._seeded))
            severity = Severity.WARNING
            message = (
                f"This run draws from {drawn} without seeding it — {already} is seeded, but "
                f"the generators are independent, so the run still varies between "
                f"invocations."
            )
        else:
            severity = Severity.NOTE
            message = (
                f"This training run draws from {drawn} with no seed set, so it cannot be "
                f"reproduced and a change in results cannot be attributed to a change in "
                f"the code."
            )

        # Report against the first unseeded draw. `report` stores the node and the engine
        # resolves the position later, which is the whole point of the deferred-position
        # design -- most files produce no findings and resolving costs more than parsing.
        self.report(
            node,
            message,
            hint=f"Call {calls} at startup, or one helper that covers every generator "
                 f"(`seed_everything(seed)` / `set_seed(seed)`).",
            severity=severity,
        )
