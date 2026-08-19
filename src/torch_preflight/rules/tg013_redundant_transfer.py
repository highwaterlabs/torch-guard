"""TG013 - a host-to-device transfer repeated every iteration."""

from __future__ import annotations

from typing import List, Optional

import libcst as cst

from ..analysis.helpers import base_name, dotted_name, final_attr, positional_args
from ..analysis.scope import target_names
from ..diagnostics import Category, Severity
from .base import Rule, register

TRANSFER_METHODS = {"to", "cuda"}

#: ``torch.*`` factories that allocate on the host unless told otherwise. Constructing one
#: inside the loop and then transferring it copies the same bytes every iteration.
HOST_FACTORIES = {
    "tensor", "zeros", "ones", "full", "empty", "arange", "linspace", "eye",
    "as_tensor", "from_numpy", "randn", "rand", "randint",
}

#: These must stay inside the loop -- the point is a fresh draw each iteration -- so the
#: fix is ``device=``, not hoisting.
RANDOM_FACTORIES = {"randn", "rand", "randint"}


def _comprehension_targets(node: cst.CSTNode) -> set:
    """Names bound by a comprehension's ``for`` clauses, including nested ones."""
    names = set()
    comp = getattr(node, "for_in", None)
    while comp is not None:
        target = getattr(comp, "target", None)
        if target is not None:
            names.update(target_names(target))
        comp = getattr(comp, "inner_for_in", None)
    return names


@register
class RedundantDeviceTransfer(Rule):
    code = "TG013"
    name = "redundant-device-transfer"
    summary = "Host-to-device transfer repeated every iteration"
    severity = Severity.WARNING
    category = Category.PERFORMANCE_WARN
    explanation = """
Moving each batch to the device inside the loop is correct and necessary — that is not what
this rule is about. It fires on transfers of data that does not change between iterations,
which pay the same host-to-device copy on every step.

Three shapes, all of them genuinely repeated work:

* ``mask = full_mask.to(device)`` where ``full_mask`` lives outside the loop and stays on
  the host. Every iteration copies it again. Hoist the transfer above the loop.
* ``torch.tensor([...]).to(device)`` inside the loop. The tensor is built on the host and
  copied each step; build it once, outside.
* ``model.to(device)`` inside the loop. This one does not re-copy — but ``Module.to`` walks
  every parameter and buffer to check them, which for a large model is thousands of Python
  calls per step for no effect.

**Not flagged**, because these cost nothing:

* ``images.to(device)`` where ``images`` comes from the loop itself — a real batch transfer.
* ``x = x.to(device)`` re-assigning the same name. ``Tensor.to`` returns ``self`` when the
  tensor is already on the requested device, so only the first iteration copies anything.

Use ``non_blocking=True`` with a pinned-memory DataLoader for the batch transfers that do
belong in the loop, so the copy overlaps with compute (see TG004).
""".strip()

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        #: Names being assigned right now, so ``x = x.to(...)`` can be recognised as the
        #: cheap self-assignment it is.
        self._assigning: List[str] = []

    def visit_Assign(self, node: cst.Assign) -> bool:
        for target in node.targets:
            name = dotted_name(target.target)
            if name:
                self._assigning.append(name)
        return True

    def leave_Assign(self, original_node: cst.Assign) -> None:
        self._assigning.clear()

    def visit_Call(self, node: cst.Call) -> bool:
        func = node.func
        if not isinstance(func, cst.Attribute) or func.attr.value not in TRANSFER_METHODS:
            return True
        if not self.in_loop:
            return True
        device = self._device_argument(node)
        if device is None:
            return True
        # ``clip_coef.to(device)`` inside ``for device, grads in ...`` cannot be hoisted:
        # the destination changes each iteration. Found in torch's own clip_grad.
        if self._is_loop_bound(dotted_name(device)):
            return True

        receiver = func.value

        # Restoring the device after a deliberate `.cpu()` is required, not redundant.
        # `fast_neural_style` does `transformer.eval().cpu()`, saves a checkpoint, then
        # `transformer.to(device).train()`. Hoisting that out would leave the model on the
        # host for the rest of training.
        if self._moved_to_host_in_loop(receiver):
            return True

        # A tensor built on the host inside the loop, then copied to the device.
        factory = self._host_factory(receiver)
        if factory is not None:
            # A random factory must stay in the loop -- the point is fresh values each
            # iteration -- so "hoist it" would be wrong advice. Only the double allocation
            # is the problem there.
            if factory in RANDOM_FACTORIES:
                hint = ("Pass `device=` to the factory so it is allocated on the device "
                        "directly, instead of built on the host and copied.")
            else:
                hint = ("Construct it once outside the loop, or pass `device=` to the "
                        "factory so it is allocated on the device directly.")
            self.report(
                node,
                f"`torch.{factory}(...)` is built on the host inside this loop and copied "
                f"to the device every iteration.",
                hint=hint,
            )
            return True

        name = dotted_name(receiver)
        if not name:
            return True

        # The batch itself: bound by the loop, so this is a real per-step transfer. The
        # *root* is what matters -- ``for shard in shards: shard.tensor.to(...)`` binds
        # ``shard``, not ``shard.tensor``, and checking the dotted name flagged every one of
        # those in torch's sharded-tensor code.
        if self._is_loop_bound(name):
            return True

        # ``x = x.to(device)`` is a no-op after the first iteration, because ``Tensor.to``
        # returns ``self`` when the tensor already lives on the target device.
        if name in self._assigning:
            return True

        if self.prov.is_model(name, self.scope_path):
            self.report(
                node,
                f"`{name}` is a model and this moves it to the device on every iteration. "
                f"`Module.to()` walks every parameter and buffer each time, even when they "
                f"are already there.",
                hint=f"Move `{name}.to(device)` above the loop.",
            )
            return True

        self.report(
            node,
            f"`{name}` does not change inside this loop, so this copies the same data to "
            f"the device on every iteration.",
            hint=f"Hoist the transfer above the loop and reuse the result.",
        )
        return True

    # ------------------------------------------------------------------ helpers

    def _is_loop_bound(self, name: Optional[str]) -> bool:
        """Is this name, or the object it is an attribute of, bound by an enclosing loop?

        Comprehensions count. ``[t.cuda(rank) for t in tensors]`` iterates just as a ``for``
        does, but the target is bound in a comprehension scope rather than a ``LoopFrame``,
        so ``bound_in_any_loop`` alone flagged every one of those in torch's distributed
        tests.
        """
        if not name:
            return False
        root = name.split(".")[0]
        if self.bound_in_any_loop(name) or self.bound_in_any_loop(root):
            return True
        return self._bound_by_comprehension(root)

    def _bound_by_comprehension(self, root: str) -> bool:
        for scope in self.scopes:
            if scope.kind != "comprehension" or scope.node is None:
                continue
            if root in _comprehension_targets(scope.node):
                return True
        return False

    def _device_argument(self, node: cst.Call) -> Optional[cst.BaseExpression]:
        """The device being moved to, or None if this is not clearly a device move.

        ``.to()`` is overloaded: ``x.to(device)``, ``x.to(dtype)``, ``x.to(other_tensor)``.
        Guessing costs precision in both directions -- reading ``x.to(dtype)`` as a transfer
        produced false positives in torch's own FSDP code -- so this requires *positive*
        evidence and returns None otherwise. A missed transfer is a quiet warning we did not
        emit; a misread cast is a wrong one we did.
        """
        if final_attr(node.func) == "cuda":
            return node  # `.cuda()` is unambiguous; the node stands in for the device
        for arg in node.args:
            if arg.keyword is not None and arg.keyword.value == "device":
                return arg.value
        args = positional_args(node)
        if not args:
            return None
        first = args[0].value
        if isinstance(first, cst.SimpleString):
            text = first.value.strip("\"'").lower()
            # `.to("cpu")` is a *download*, and this rule is about re-uploading the same data
            # to the device every iteration. `pinmem_nonblock.py` -- a tutorial whose subject
            # is measuring transfer behaviour -- loops 100 times over
            # `tensor.to("cpu", non_blocking=True)` on a tensor created with `device="cuda"`,
            # which was wrong on both counts.
            if "cpu" in text:
                return None
            return first if ("cuda" in text or "mps" in text) else None
        name = dotted_name(first)
        if name and "device" in name.lower():
            return first
        return None

    def _moved_to_host_in_loop(self, receiver: cst.BaseExpression) -> bool:
        """Is this receiver explicitly sent to the host somewhere in the same loop?"""
        base = base_name(receiver)
        frame = self.innermost_loop
        if base is None or frame is None:
            return False

        class _Probe(cst.CSTVisitor):
            def __init__(self) -> None:
                self.found = False

            def visit_Call(self, node: cst.Call) -> bool:
                func = node.func
                if not isinstance(func, cst.Attribute):
                    return True
                if func.attr.value == "cpu" and base_name(func.value) == base:
                    self.found = True
                return True

        probe = _Probe()
        frame.node.visit(probe)
        return probe.found

    def _host_factory(self, receiver: cst.BaseExpression) -> Optional[str]:
        """``torch.zeros(...)`` etc., built on the host because no ``device=`` was given."""
        if not isinstance(receiver, cst.Call):
            return None
        leaf = final_attr(receiver.func)
        if leaf not in HOST_FACTORIES:
            return None
        dotted = dotted_name(receiver.func) or ""
        if not (dotted.startswith("torch.") or dotted == leaf):
            return None
        # An explicit device= already puts it in the right place; the ``.to`` is redundant
        # but harmless, and flagging it would be noise.
        for arg in receiver.args:
            if arg.keyword is not None and arg.keyword.value == "device":
                return None
        return leaf
