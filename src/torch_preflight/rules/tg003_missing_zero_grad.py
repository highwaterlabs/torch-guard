"""TG003 - ``.backward()`` inside a loop with no ``zero_grad()`` anywhere in that loop."""

from __future__ import annotations

from typing import Optional

import libcst as cst

from ..analysis.helpers import contains_call_to, dotted_name
from ..diagnostics import Category, Severity
from .base import Rule, register

#: Frameworks that own the optimizer step and clear gradients for you, plus
#: ``dist_autograd``, whose ``backward(context_id, [loss])`` accumulates into an RPC
#: context rather than into ``.grad`` — ``zero_grad()`` is meaningless there.
MANAGED_BACKWARD_OWNERS = ("lightning", "fabric", "dist_autograd")

#: Receivers whose ``.step()`` means "apply the accumulated gradients".
OPTIMIZER_HINTS = ("optim", "opt", "scaler", "accelerator")


class _OptimizerStepFinder(cst.CSTVisitor):
    """Looks for an optimizer step, which is what makes stale gradients matter.

    ``.backward()`` in a loop is only a bug if something then *applies* those gradients.
    Raw autograd over a tensor recreated each iteration — common in tests and in manual
    gradient computation — accumulates nothing, because the leaf is new every time.
    """

    def __init__(self) -> None:
        self.found = False

    def visit_Call(self, node: cst.Call) -> Optional[bool]:
        func = node.func
        if not isinstance(func, cst.Attribute) or func.attr.value != "step":
            return True
        receiver = (dotted_name(func.value) or "").lower().rsplit(".", 1)[-1]
        # ``scheduler.step()`` advances the learning rate; it applies no gradients.
        if "sched" in receiver:
            return True
        if any(hint in receiver for hint in OPTIMIZER_HINTS):
            self.found = True
        return True


def _applies_gradients(tree: cst.CSTNode) -> bool:
    finder = _OptimizerStepFinder()
    tree.visit(finder)
    return finder.found


@register
class MissingZeroGrad(Rule):
    code = "TG003"
    name = "missing-zero-grad"
    summary = "backward() in a loop without a matching zero_grad()"
    severity = Severity.ERROR
    category = Category.CONVERGENCE_BUG
    explanation = """
PyTorch *accumulates* into ``.grad`` rather than overwriting it. Without a
``zero_grad()`` somewhere in the loop, step N's update uses the summed gradients of
steps 1..N. The run does not crash - it silently trains on a growing, wrongly scaled
gradient, so the loss curve looks plausible while the model converges to the wrong place
or diverges. This is one of the most expensive bugs in ML precisely because it is silent.

Call ``optimizer.zero_grad(set_to_none=True)`` at the top of each iteration (or after
``optimizer.step()``). Deliberate gradient accumulation still calls it, just every N
steps - a ``zero_grad()`` inside an ``if`` in the loop body counts and is not flagged.

The rule also requires an optimizer step in the loop. Calling ``.backward()`` without one
computes gradients but never applies them, so there is nothing to go stale - that pattern
shows up in tests and in manual gradient computation, and flagging it is noise.
""".strip()

    def visit_Call(self, node: cst.Call) -> bool:
        func = node.func
        if not isinstance(func, cst.Attribute) or func.attr.value != "backward":
            return True

        # Lightning/Fabric clear gradients on your behalf.
        if self.ctx.is_lightning:
            return True
        owner = (dotted_name(func.value) or "").lower()
        if any(marker in owner for marker in MANAGED_BACKWARD_OWNERS):
            return True

        loop = self.innermost_loop
        if loop is None:
            # A one-shot backward outside any loop cannot accumulate across steps.
            return True

        # Search the whole innermost loop body, so gradient accumulation guarded by an
        # ``if (i + 1) % accum == 0:`` still counts as clearing gradients.
        if contains_call_to(loop.node, ["zero_grad"]):
            return True

        # No optimizer step means nothing consumes the gradients, so nothing goes stale.
        if not _applies_gradients(loop.node):
            return True

        self.report(
            node,
            "`.backward()` runs in this loop but nothing calls `zero_grad()`; gradients "
            "accumulate across iterations and every step trains on the running sum.",
            hint="Add `optimizer.zero_grad(set_to_none=True)` at the start of the loop "
            "body (or right after `optimizer.step()`).",
        )
        return True
