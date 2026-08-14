"""TG011 - ``model.eval()`` in an epoch loop that never switches back to train mode."""

from __future__ import annotations

from typing import List, Optional

import libcst as cst

from ..analysis.helpers import contains_call_to, dotted_name
from ..diagnostics import Category, Severity
from .base import Rule, register

#: Iterables that mark a validation pass.
EVAL_ITERABLE_HINTS = ("val", "valid", "eval", "test", "dev")


class _TrainModeFinder(cst.CSTVisitor):
    """Looks for ``<receiver>.train()`` — the call that undoes ``eval()``.

    Matching the receiver matters. ``model.backbone.eval()`` to freeze batch-norm during
    fine-tuning is deliberate and is *not* undone by ``model.train()``; treating them as a
    pair would hide the real bug in one direction and invent one in the other.
    """

    def __init__(self, receiver: Optional[str]) -> None:
        self.receiver = receiver
        self.found = False

    def visit_Call(self, node: cst.Call) -> bool:
        func = node.func
        if not isinstance(func, cst.Attribute) or func.attr.value != "train":
            return True
        if self.receiver is None:
            self.found = True
            return True
        called_on = dotted_name(func.value)
        # ``model.train()`` restores ``model.eval()``; it also restores a submodule's, since
        # ``train()`` recurses, so an exact match or a prefix of the eval receiver counts.
        if called_on and (called_on == self.receiver
                          or self.receiver.startswith(called_on + ".")):
            self.found = True
        return True


def _restores_train_mode(tree: cst.CSTNode, receiver: Optional[str]) -> bool:
    finder = _TrainModeFinder(receiver)
    tree.visit(finder)
    return finder.found


class _EvalIterationFinder(cst.CSTVisitor):
    """Is there a loop over something that looks like a validation set?"""

    def __init__(self) -> None:
        self.found = False

    def visit_For(self, node: cst.For) -> bool:
        name = (dotted_name(node.iter) or "").lower()
        if any(hint in name for hint in EVAL_ITERABLE_HINTS):
            self.found = True
        return True


def _has_evaluation_pass(tree: cst.CSTNode) -> bool:
    finder = _EvalIterationFinder()
    tree.visit(finder)
    return finder.found


@register
class StuckInEvalMode(Rule):
    code = "TG011"
    name = "stuck-in-eval-mode"
    summary = "eval() in an epoch loop with no matching train()"
    severity = Severity.ERROR
    category = Category.CONVERGENCE_BUG
    explanation = """
``model.eval()`` is sticky. It switches dropout off and makes batch-norm use its running
statistics instead of the batch's, and it stays that way until something calls
``model.train()``.

In the usual epoch loop — train, then validate — a missing ``model.train()`` means only the
**first** epoch trains properly. From epoch two onward, dropout is disabled and batch-norm
stops updating its running statistics while still normalising with the stale ones. Training
continues, the loss keeps moving, and the model quietly underfits or overfits depending on
which layer dominated. Nothing raises.

Call ``model.train()`` at the top of each epoch, not once before the loop::

    for epoch in range(epochs):
        model.train()                 # here, every epoch
        for batch in train_loader:
            ...
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                ...

The rule only fires when it can see the whole pattern in one loop: a backward pass, a
validation iteration, an ``eval()`` and no matching ``train()``. Deliberately freezing a
submodule (``model.backbone.eval()`` for small-batch fine-tuning) is not flagged, because
the receiver has to match. An ``evaluate()`` helper in another function is not flagged
either — restoring train mode is the caller's job and that is not visible here.
""".strip()

    def visit_Call(self, node: cst.Call) -> bool:
        func = node.func
        if not isinstance(func, cst.Attribute) or func.attr.value != "eval":
            return True
        if self.ctx.is_lightning:
            return True  # Lightning owns the mode switches

        loops: List = self.loops
        if not loops:
            # A one-shot eval outside any loop never repeats, so nothing is stuck.
            return True

        receiver = dotted_name(func.value)
        # The epoch loop is the outermost one: that is the scope a per-epoch `train()`
        # would have to live in for the second epoch to train correctly.
        epoch_loop = loops[0].node

        # Require the full shape, so this cannot fire on an eval-only script.
        if not contains_call_to(epoch_loop, ["backward"]):
            return True
        if not _has_evaluation_pass(epoch_loop):
            return True
        if _restores_train_mode(epoch_loop, receiver):
            return True

        subject = f"`{receiver}.eval()`" if receiver else "`eval()`"
        self.report(
            node,
            f"{subject} is called in this loop but nothing calls "
            f"`{receiver or 'model'}.train()` in it, so every epoch after the first trains "
            f"with dropout disabled and batch-norm frozen on stale statistics.",
            hint=f"Call `{receiver or 'model'}.train()` at the top of the epoch loop, not "
                 f"once before it.",
        )
        return True
