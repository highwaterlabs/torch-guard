"""TG007 - a GPU synchronisation point inside the inner loop."""

from __future__ import annotations

from typing import Optional

import libcst as cst

from ..analysis.helpers import contains_call_to, dotted_name, final_attr
from ..analysis.scope import target_names
from ..diagnostics import Category, Severity
from .base import Rule, register

#: Calls that force the GPU to finish and hand a value back to Python. Each one drains the
#: pipeline: the CPU stops issuing kernels until the device catches up.
SYNC_METHODS = {"item", "cpu", "numpy", "tolist"}

#: Iterables that mean "this loop yields batches, not elements". One sync per batch is a
#: normal logging granularity -- `print(val_loss.item())` in a validation loop is not the
#: per-element thrashing this rule is about.
BATCH_ITERABLES = ("loader", "dataloader", "dataset", "batches", "loaders", "iterator")


@register
class DeviceSyncInInnerLoop(Rule):
    code = "TG007"
    name = "device-sync-in-inner-loop"
    summary = "GPU synchronisation inside a loop nested in the training step"
    severity = Severity.WARNING
    category = Category.PERFORMANCE_WARN
    explanation = """
CUDA kernels are queued asynchronously: Python races ahead issuing work while the GPU
executes it. ``.item()``, ``.cpu()``, ``.numpy()`` and ``.tolist()`` each need a real value,
so they block until the queue drains — and while it drains, nothing is being issued.

**One sync per training step is fine, and this rule does not flag it.** ``loss.item()`` once
an iteration is exactly what TG001 tells you to write, and its cost is a rounding error
against a forward and backward pass. Flagging it would contradict a rule this tool already
ships.

What costs real time is syncing *per element*: a sync inside a loop that is itself inside the
training step, so it runs many times per iteration instead of once::

    for batch in loader:                  # the training step
        loss = criterion(model(batch), y)
        loss.backward()
        for i in range(len(preds)):       # nested loop
            correct += preds[i].item()    # one full pipeline drain per element

That drains the pipeline once per element. Keep the work on the device and sync once::

    correct += (preds == targets).sum().item()

``torch.cuda.synchronize()`` inside the training loop is flagged for the same reason: it is
an unconditional drain every step, and it is almost always left over from timing code.

Comprehensions count as loops, because ``[p.item() for p in preds]`` syncs once per element
exactly as the explicit loop does.
""".strip()

    def visit_Call(self, node: cst.Call) -> bool:
        if self._is_cuda_synchronize(node):
            if self._inside_training_loop():
                self.report(
                    node,
                    "`torch.cuda.synchronize()` runs inside the training loop, draining the "
                    "kernel queue every step so the CPU cannot stay ahead of the GPU.",
                    hint="Remove it, or guard it behind a timing flag. Kernels are already "
                         "ordered; explicit synchronisation is only needed to measure them.",
                )
            return True

        func = node.func
        if not isinstance(func, cst.Attribute) or func.attr.value not in SYNC_METHODS:
            return True

        method = func.attr.value
        # `.numpy()` on a plain ndarray, `.tolist()` on a list: no device involved. Requiring
        # a grad-bearing or model-derived receiver keeps this to real tensors.
        receiver = dotted_name(func.value)
        if not self._looks_like_a_tensor(func.value, receiver):
            return True

        nested = self._nested_loop_inside_training_step()
        if nested is None:
            return True

        subject = f"`{receiver}.{method}()`" if receiver else f"`.{method}()`"
        self.report(
            node,
            f"{subject} synchronises with the GPU inside a loop nested in the training "
            f"step, so the kernel queue drains once per element rather than once per step.",
            hint="Do the reduction on the device and sync once — `(preds == targets)"
                 ".sum().item()` rather than a Python loop over elements.",
        )
        return True

    # ------------------------------------------------------------------ helpers

    def _is_cuda_synchronize(self, node: cst.Call) -> bool:
        return (dotted_name(node.func) or "").endswith("cuda.synchronize")

    def _inside_training_loop(self) -> bool:
        return any(contains_call_to(frame.node, ["backward"]) for frame in self.loops)

    def _in_comprehension(self) -> bool:
        return any(scope.kind == "comprehension" for scope in self.scopes)

    def _iterates_batches(self, frame) -> bool:
        iterable = (frame.iterable or "").lower()
        return any(hint in iterable for hint in BATCH_ITERABLES)

    def _nested_loop_inside_training_step(self):
        """The innermost loop, if it is nested inside a loop that does the backward pass.

        The distinction that keeps this from contradicting TG001: the training step is the
        loop containing ``.backward()``. A sync directly in *that* loop happens once per
        step and is fine. A sync in a loop *inside* it happens many times per step.
        """
        loops = self.loops
        if not loops:
            return None

        # A comprehension iterates without creating a LoopFrame, so
        # `[p.item() for p in preds]` inside the training loop is per-element syncing even
        # though only one loop is on the stack.
        if self._in_comprehension() and self._inside_training_loop():
            return loops[-1]

        if len(loops) < 2:
            return None
        innermost = loops[-1]
        # If the innermost loop is itself the one doing the backward pass, the sync is
        # once per step.
        if contains_call_to(innermost.node, ["backward"]):
            return None
        # A loop over a dataloader yields *batches*. One sync per batch is the granularity
        # people log at, and flagging `print(val_loss.item())` in a validation loop would be
        # noise -- the rule is about syncing per element.
        if self._iterates_batches(innermost):
            return None
        outer_does_backward = any(
            contains_call_to(frame.node, ["backward"]) for frame in loops[:-1]
        )
        return innermost if outer_does_backward else None

    def _looks_like_a_tensor(self, node: cst.BaseExpression, name: Optional[str]) -> bool:
        """Is the receiver plausibly a device tensor?

        A chained call like ``preds.detach().cpu()`` has no dotted name, but the chain itself
        is tensor-shaped, so those are accepted. Bare names are accepted only when
        provenance says they carry a graph or came from a model, which keeps numpy arrays and
        plain lists out.
        """
        if isinstance(node, cst.Call):
            return final_attr(node.func) in {
                "detach", "clone", "float", "half", "squeeze", "view", "reshape",
                "argmax", "sum", "mean", "softmax", "sigmoid", "max", "min",
            }
        if isinstance(node, cst.Subscript):
            base = dotted_name(node.value)
            return bool(base) and self._known_tensor(base)
        return bool(name) and self._known_tensor(name)

    def _known_tensor(self, name: str) -> bool:
        if self.prov.is_grad_name(name, self.scope_path):
            return True
        if self.prov.is_model(name, self.scope_path):
            return True
        if self._names_a_tensor(name):
            return True
        # A comprehension target is an *element* of whatever it iterates, and nothing tracks
        # it. `[p.item() for p in preds]` has to be judged on `preds`, not on `p`.
        iterated = self._comprehension_source(name)
        return bool(iterated) and (
            self.prov.is_grad_name(iterated, self.scope_path)
            or self._names_a_tensor(iterated)
        )

    def _names_a_tensor(self, name: str) -> bool:
        """The naming convention fallback: loop variables carry no provenance."""
        leaf = name.rsplit(".", 1)[-1].lower()
        return any(hint in leaf for hint in (
            "pred", "logit", "output", "out", "target", "label", "score", "prob", "loss",
        ))

    def _comprehension_source(self, name: str) -> Optional[str]:
        """If ``name`` is a comprehension target, the dotted name it iterates over."""
        for scope in self.scopes:
            if scope.kind != "comprehension" or scope.node is None:
                continue
            comp = getattr(scope.node, "for_in", None)
            while comp is not None:
                target = getattr(comp, "target", None)
                if target is not None and name in target_names(target):
                    return dotted_name(getattr(comp, "iter", None))
                comp = getattr(comp, "inner_for_in", None)
        return None
