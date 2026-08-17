"""TG001 - storing a graph-attached tensor in a container that outlives the iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import libcst as cst

from ..analysis.helpers import attach_method, dotted_name, final_attr
from ..analysis.scope import ScopePath
from ..diagnostics import Category, Severity
from .base import Rule, register

ACCUMULATING_METHODS = {"append", "add", "extend", "insert", "put", "push"}


@dataclass
class _Candidate:
    """A finding held back until the whole module has been read."""

    node: cst.CSTNode
    #: The container (or accumulator) whose retention is in question.
    holder: str
    #: Scope key the holder belongs to; ``None`` for ``self.*`` instance state.
    scope: Optional[ScopePath]
    message: str
    hint: str


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

**Deferred backward is not a leak, and this rule does not flag it.** Some code stores
graph-attached tensors precisely so a backward pass can run over them later, and detaching
would break it::

    def _maybe_compute_loss(self, stage, output, target_mbs, mb_index):
        loss = self._compute_loss(output, target_mbs[mb_index])
        self._internal_losses.append(loss)      # required, not a leak

    def _maybe_get_loss(self, stage, mb_index):
        return self._internal_losses[mb_index]  # the graph is still needed here

That is how pipeline parallelism schedules microbatches: the loss for each microbatch is
computed on one rank, held, and backwarded when the schedule reaches it. The same shape
appears in ``torch.distributed.autograd``, where a chain of intermediate tensors is kept so
``dist_autograd.backward(context_id, [res[i].sum()])`` can traverse it.

So a container is exempt once an element read out of it reaches a backward pass — directly,
via a backward-taking call, or by being returned to a caller who does the backward. The
retention is then load-bearing, and the only honest advice is none.
""".strip()

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._candidates: List[_Candidate] = []
        #: Holders whose contents a backward pass still needs. See ``_note_backward_feed``.
        self._backward_fed: Set[Tuple[Optional[ScopePath], str]] = set()

    # ------------------------------------------------------------------ collection

    def visit_Call(self, node: cst.Call) -> bool:
        # A backward pass reads whatever it is handed, so record those holders before
        # considering any candidate -- a call can be both, as in `losses[i].backward()`.
        if self._is_backward_call(node):
            for arg in node.args:
                self._note_backward_feed(arg.value)
            if isinstance(node.func, cst.Attribute):
                self._note_backward_feed(node.func.value)
            return True

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

        self._defer(
            stored,
            container,
            f"`{container}.{method}(...)` stores a tensor that is still attached to the "
            f"autograd graph; every iteration's graph is retained in VRAM.",
            "Use `.item()` to keep just the scalar value, or `.detach()` to keep the "
            "tensor without its graph.",
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

        self._defer(
            node.value,
            target,
            f"`{target} += ...` accumulates a graph-attached tensor across iterations, "
            f"chaining every step's graph into one that is never freed.",
            "Accumulate the scalar instead: `{0} += loss.item()`.".format(target),
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
            self._defer(
                node.value,
                container,
                f"`{container}[...] = ...` stores a graph-attached tensor in a container "
                f"that outlives the loop iteration, retaining its graph in VRAM.",
                "Store `.item()` or `.detach()` instead.",
            )
            break
        return True

    def visit_Return(self, node: cst.Return) -> bool:
        # Handing a raw element out of the container gives the caller something we cannot
        # follow. `return self._internal_losses[i]` is a getter for a deferred backward;
        # `return loss` is not, so only subscript reads count here.
        if node.value is not None:
            self._note_escaping_reads(node.value)
        return True

    # ------------------------------------------------------------------ verdict

    def leave_Module(self, original_node: cst.Module) -> None:
        for candidate in self._candidates:
            if (candidate.scope, candidate.holder) in self._backward_fed:
                continue
            self.report(
                candidate.node,
                candidate.message,
                hint=candidate.hint,
                fix_build=lambda updated: attach_method(updated, "detach"),
                fix_description="add .detach()",
            )

    # ------------------------------------------------------------------ helpers

    def _defer(self, node: cst.CSTNode, holder: str, message: str, hint: str) -> None:
        """Hold a finding until ``leave_Module``.

        The exemption depends on code that may appear anywhere in the file — a getter
        defined below the append, or a backward call in a sibling method — so the verdict
        cannot be reached at the point of the write.
        """
        self._candidates.append(
            _Candidate(node, holder, self._holder_scope(holder), message, hint)
        )

    def _holder_scope(self, holder: str) -> Optional[ScopePath]:
        """Which scope a holder's identity is keyed to.

        ``self.losses`` is instance state: appended in one method and backwarded in
        another is the normal shape, so it matches file-wide. A bare local name matches
        only within its own function — the file-wide leakage that has already bitten
        ``models``, ``criteria``, ``uses_distributed`` and TG008.
        """
        if holder.startswith("self."):
            return None
        return self._function_key()

    def _function_key(self) -> ScopePath:
        """Scope path down to the innermost enclosing function."""
        last = 0
        for index, scope in enumerate(self.scopes):
            if scope.kind == "function":
                last = index
        return tuple(s.name for s in self.scopes[: last + 1])

    def _is_backward_call(self, node: cst.Call) -> bool:
        """A call that runs a backward pass over whatever it is given.

        Matched on the callee's last segment containing ``backward``, which covers
        ``loss.backward()``, ``torch.autograd.backward(...)``,
        ``dist_autograd.backward(ctx, [...])`` and pipelining's ``backward_one_chunk(...)``.
        """
        leaf = final_attr(node.func)
        return leaf is not None and "backward" in leaf

    def _note_backward_feed(self, node: cst.BaseExpression) -> None:
        """Record every holder read anywhere inside an expression fed to a backward pass.

        The read can be arbitrarily deep: ``dist_autograd.backward(ctx, [res[i].sum()])``
        reaches ``res`` through a list, a call and a subscript.
        """
        for name in _reads(node, subscripts_only=False):
            self._mark_fed(name)

    def _note_escaping_reads(self, node: cst.BaseExpression) -> None:
        for name in _reads(node, subscripts_only=True):
            self._mark_fed(name)

    def _mark_fed(self, name: str) -> None:
        self._backward_fed.add((self._holder_scope(name), name))
        # A loop variable is an element of what it iterates, so `for l in self.losses:
        # l.backward()` has to mark the container, not the throwaway name.
        for frame in self.loops:
            if name in frame.assigned and frame.iterable:
                iterable = frame.iterable
                self._backward_fed.add((self._holder_scope(iterable), iterable))

    def _container_accumulates(self, container: str) -> bool:
        """True if writes to ``container`` survive past the current loop iteration."""
        if self.in_loop:
            # A list built fresh each iteration cannot accumulate across iterations.
            return not self.bound_in_innermost_loop(container)
        # Outside a loop, only instance state persists across repeated calls
        # (``self.outputs.append(...)`` in a step method is the classic Lightning leak).
        return container.startswith("self.")


def _reads(node: cst.CSTNode, *, subscripts_only: bool) -> List[str]:
    """Dotted names read within an expression.

    With ``subscripts_only``, only the base of a subscript is collected: ``losses[i]``
    yields ``losses`` but a bare ``loss`` yields nothing. That distinction is what keeps
    ``return loss`` (the Lightning leak, where the container is still write-only) apart
    from ``return self._internal_losses[i]`` (a getter for a held graph).
    """
    collector = _ReadCollector(subscripts_only)
    node.visit(collector)
    return collector.names


class _ReadCollector(cst.CSTVisitor):
    def __init__(self, subscripts_only: bool) -> None:
        self.subscripts_only = subscripts_only
        self.names: List[str] = []

    def visit_Subscript(self, node: cst.Subscript) -> bool:
        name = dotted_name(node.value)
        if name is not None:
            self.names.append(name)
        return True

    def visit_Name(self, node: cst.Name) -> bool:
        if not self.subscripts_only:
            self.names.append(node.value)
        return True

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        name = dotted_name(node)
        if name is None:
            return True  # ``f().x`` -- descend, the reads are inside the call
        if not self.subscripts_only:
            self.names.append(name)
        # A pure name path contains no further reads. Descending would collect its own
        # segments, so ``self.losses`` in a backward call would also mark a local
        # ``losses`` in some other function as load-bearing.
        return False
