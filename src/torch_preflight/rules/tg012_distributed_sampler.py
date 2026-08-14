"""TG012 - a DataLoader under DDP with no DistributedSampler."""

from __future__ import annotations

from typing import List, Optional

import libcst as cst

from ..analysis.helpers import dotted_name, final_attr, keyword_arg
from ..analysis.scope import target_names
from ..diagnostics import Category, Severity
from .base import Rule, register

#: Frameworks that inject a distributed sampler for you. Lightning wraps loaders in
#: ``_get_distributed_sampler``, Accelerate's ``prepare`` re-creates them with a shard-aware
#: sampler, and the HF ``Trainer`` builds its own. Flagging these would be wrong twice over:
#: the code is already correct, and adding a sampler on top double-shards the data.
SAMPLER_INJECTORS = ("lightning", "fabric", "accelerate", "accelerator", "trainer")

#: Names suggesting the loader feeds evaluation rather than training.
EVAL_HINTS = ("val", "valid", "eval", "test")

#: Calls that mean "this code path runs on more than one rank".
DISTRIBUTED_MARKERS = ("DistributedDataParallel", "DDP", "init_process_group")


class _DistributedMarkerFinder(cst.CSTVisitor):
    def __init__(self) -> None:
        self.found = False

    def visit_Call(self, node: cst.Call) -> bool:
        if final_attr(node.func) in DISTRIBUTED_MARKERS:
            self.found = True
        return True


def _sets_up_distributed(tree: cst.CSTNode) -> bool:
    finder = _DistributedMarkerFinder()
    tree.visit(finder)
    return finder.found


@register
class MissingDistributedSampler(Rule):
    code = "TG012"
    name = "missing-distributed-sampler"
    summary = "DataLoader under DDP without a DistributedSampler"
    severity = Severity.ERROR
    category = Category.CONVERGENCE_BUG
    explanation = """
``DistributedSampler`` is what splits the dataset across ranks. Without it every rank
iterates the **whole** dataset in the same order, so all N ranks compute gradients on
identical batches. DDP then averages those identical gradients, which changes nothing.

The result is a run that costs N GPUs and trains as if it had one, on 1/N of the data you
think you are using per epoch. It does not crash, throughput looks right, and the loss
curve is a plausible single-GPU curve — so the usual conclusion is that multi-GPU "isn't
scaling well" rather than that the data is duplicated.

Build the loader with a sampler instead of ``shuffle``, since they are mutually exclusive::

    sampler = DistributedSampler(dataset)          # shuffles within each rank's shard
    loader = DataLoader(dataset, sampler=sampler)  # no shuffle= here

and call ``sampler.set_epoch(epoch)`` each epoch, or the shuffle order never changes.

Evaluation loaders are reported as warnings rather than errors: duplicating validation
across ranks wastes work but computes the right number, and evaluating on one rank is a
legitimate choice.

Lightning, Accelerate and the Hugging Face ``Trainer`` inject a sampler themselves, so the
rule stays silent when it can see one of them.
""".strip()

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        #: Names being assigned right now. The dispatcher runs one traversal in source
        #: order, so an enclosing ``Assign`` is entered before the call on its right-hand
        #: side -- which is how ``val_loader = DataLoader(...)`` is recognised as feeding
        #: evaluation.
        self._assigning: List[str] = []
        #: Cached: does module-level code set up distributed training?
        self._module_distributed: Optional[bool] = None

    def visit_Assign(self, node: cst.Assign) -> bool:
        for target in node.targets:
            self._assigning.extend(target_names(target.target))
        return True

    def leave_Assign(self, original_node: cst.Assign) -> None:
        self._assigning.clear()

    def visit_Call(self, node: cst.Call) -> bool:
        if final_attr(node.func) != "DataLoader":
            return True
        if not self.ctx.uses_distributed or self.ctx.is_lightning:
            return True

        # The file-level fact is only a cheap pre-filter. Firing on it alone flags loaders
        # in functions that have nothing to do with DDP, just because some *other* function
        # in the file sets up a process group -- the same file-wide leakage that made
        # `prov.models` and `criteria` report against unrelated names. Require the marker in
        # the loader's own function, or at module level where it governs everything.
        function = self.current_function
        if function is not None and function.node is not None:
            in_scope = (
                _sets_up_distributed(function.node) or self._module_level_distributed()
            )
            if not in_scope:
                return True
        # A sampler-injecting framework anywhere in the file is enough to stay quiet: a
        # false positive here tells someone to shard data that is already sharded.
        lowered = self.ctx.source.lower()
        if any(marker in lowered for marker in SAMPLER_INJECTORS):
            return True

        if self._has_distributed_sampler(node):
            return True

        evaluation = self._looks_like_evaluation(node)
        shuffles = self._shuffles(node)

        if evaluation:
            self.report(
                node,
                "This DataLoader runs under DDP with no `DistributedSampler`, so every rank "
                "evaluates the entire dataset — N times the work for the same number.",
                hint="Pass `sampler=DistributedSampler(dataset, shuffle=False)` and reduce "
                     "the metric across ranks, or evaluate on one rank only.",
                severity=Severity.WARNING,
            )
            return True

        detail = (
            " `shuffle=True` shuffles identically on every rank, which spreads nothing."
            if shuffles else ""
        )
        self.report(
            node,
            "This DataLoader runs under DDP with no `DistributedSampler`, so every rank "
            "iterates the whole dataset and computes gradients on identical batches."
            + detail,
            hint="Pass `sampler=DistributedSampler(dataset)` instead of `shuffle=True` "
                 "(they are mutually exclusive) and call `sampler.set_epoch(epoch)`.",
        )
        return True

    # ------------------------------------------------------------------ helpers

    def _module_level_distributed(self) -> bool:
        """Is the process group set up at module level, governing every function below?

        A common shape in single-file training scripts: ``init_process_group`` runs on
        import (or under ``if __name__``) and every loader in the file is distributed.
        Only top-level statements count -- scanning the whole module would put the
        file-wide leakage straight back.
        """
        if self._module_distributed is None:
            self._module_distributed = any(
                _sets_up_distributed(statement)
                for statement in self.ctx.module.body
                if not isinstance(statement, (cst.FunctionDef, cst.ClassDef))
            )
        return self._module_distributed

    def _has_distributed_sampler(self, node: cst.Call) -> bool:
        """Is a distributed sampler passed, inline or by name?

        ``batch_sampler`` counts too: a custom batch sampler is presumed to handle
        sharding, and second-guessing it would be noise.
        """
        if keyword_arg(node, "batch_sampler") is not None:
            return True
        sampler = keyword_arg(node, "sampler")
        if sampler is None:
            return False

        value = sampler.value
        if isinstance(value, cst.Call):
            # Any sampler built inline is taken at face value except a plainly local one.
            return final_attr(value.func) != "RandomSampler"
        name = dotted_name(value)
        if name and name in self.ctx.distributed_sampler_vars:
            return True
        # A sampler passed by a name we cannot resolve: assume the author meant it.
        return name is not None

    def _looks_like_evaluation(self, node: cst.Call) -> bool:
        """Does this loader feed evaluation? Read from the name it is assigned to."""
        return any(
            hint in name.lower() for name in self._assigning for hint in EVAL_HINTS
        )

    def _shuffles(self, node: cst.Call) -> bool:
        shuffle = keyword_arg(node, "shuffle")
        return (
            shuffle is not None
            and isinstance(shuffle.value, cst.Name)
            and shuffle.value.value == "True"
        )
