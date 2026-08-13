"""TG001 - storing a graph-attached tensor in a container that outlives the iteration."""

from __future__ import annotations

from typing import Optional

import libcst as cst

from ..analysis.helpers import attach_method, dotted_name
from ..diagnostics import Category, Severity
from .base import Rule, register

ACCUMULATING_METHODS = {"append", "add", "extend", "insert", "put", "push"}


@register
class GraphRetention(Rule):
    code = "TG001"
    name = "graph-retention"
    summary = "Tensor stored with its autograd graph still attached"
    severity = Severity.ERROR
    category = Category.CRITICAL_OOM
    explanation = """
Appending a tensor that still carries ``grad_fn`` keeps its entire computational graph
alive in VRAM. The graph holds every intermediate activation produced on the way to that
tensor, so a training loop that logs one loss per step retains one full graph per step.
Memory grows linearly with iteration count and the run dies with CUDA OOM partway
through - usually after hours of GPU time.

Call ``.item()`` for scalars you only want to log, or ``.detach()`` for tensors you need
to keep as tensors.
""".strip()

    def visit_Call(self, node: cst.Call) -> bool:
        func = node.func
        if not isinstance(func, cst.Attribute):
            return True
        method = func.attr.value
        if method not in ACCUMULATING_METHODS:
            return True

        container = dotted_name(func.value)
        if container is None or not self._container_accumulates(container):
            return True

        # ``insert(index, value)`` stores its second argument.
        args = [a for a in node.args if a.keyword is None]
        if not args:
            return True
        stored = args[1].value if method == "insert" and len(args) > 1 else args[0].value

        if self.in_no_grad or not self.is_grad(stored):
            return True

        self.report(
            stored,
            f"`{container}.{method}(...)` stores a tensor that is still attached to the "
            f"autograd graph; every iteration's graph is retained in VRAM.",
            hint="Use `.item()` to keep just the scalar value, or `.detach()` to keep the "
            "tensor without its graph.",
            fix_build=lambda updated: attach_method(updated, "detach"),
            fix_description="add .detach()",
        )
        return True

    def visit_AugAssign(self, node: cst.AugAssign) -> bool:
        # ``total_loss += loss`` accumulates graphs just as surely as ``.append``.
        if not self.in_loop or self.in_no_grad:
            return True
        if not isinstance(node.operator, (cst.AddAssign, cst.SubtractAssign)):
            return True

        target = dotted_name(node.target)
        if target is None or self.bound_in_innermost_loop(target):
            return True
        if not self.is_grad(node.value):
            return True

        self.report(
            node.value,
            f"`{target} += ...` accumulates a graph-attached tensor across iterations, "
            f"chaining every step's graph into one that is never freed.",
            hint="Accumulate the scalar instead: `{0} += loss.item()`.".format(target),
            fix_build=lambda updated: attach_method(updated, "detach"),
            fix_description="add .detach()",
        )
        return True

    def visit_Assign(self, node: cst.Assign) -> bool:
        # ``cache[key] = loss`` / ``self.buffer[i] = out``
        if self.in_no_grad or not self.is_grad(node.value):
            return True

        for target in node.targets:
            if not isinstance(target.target, cst.Subscript):
                continue
            container = dotted_name(target.target.value)
            if container is None or not self._container_accumulates(container):
                continue
            self.report(
                node.value,
                f"`{container}[...] = ...` stores a graph-attached tensor in a container "
                f"that outlives the loop iteration, retaining its graph in VRAM.",
                hint="Store `.item()` or `.detach()` instead.",
                fix_build=lambda updated: attach_method(updated, "detach"),
                fix_description="add .detach()",
            )
            break
        return True

    # ------------------------------------------------------------------ helpers

    def _container_accumulates(self, container: str) -> bool:
        """True if writes to ``container`` survive past the current loop iteration."""
        if self.in_loop:
            # A list built fresh each iteration cannot accumulate across iterations.
            return not self.bound_in_innermost_loop(container)
        # Outside a loop, only instance state persists across repeated calls
        # (``self.outputs.append(...)`` in a step method is the classic Lightning leak).
        return container.startswith("self.")
