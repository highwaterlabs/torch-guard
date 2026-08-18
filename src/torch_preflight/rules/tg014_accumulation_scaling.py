"""TG014 - gradient accumulation without scaling the loss."""

from __future__ import annotations

from typing import List, Optional

import libcst as cst

from ..analysis.helpers import contains_call_to, dotted_name, final_attr
from ..diagnostics import Category, Severity
from .base import Rule, register

#: Names that mean "how many micro-batches per optimizer step".
ACCUMULATION_HINTS = ("accum", "grad_steps", "gradient_steps", "micro_batches")

#: Receivers whose ``.step()`` applies gradients. Mirrors TG003.
OPTIMIZER_HINTS = ("optim", "opt", "scaler", "accelerator")

#: Things that own the scaling for you, so the division must *not* be there.
#: ``accelerator.accumulate(...)`` and the HF ``Trainer`` both divide internally, and
#: dividing again would silently shrink the gradient by the accumulation factor.
MANAGED_ACCUMULATION = ("accelerate", "accelerator", "trainer", "lightning", "fabric",
                        "deepspeed")


def _key(node: cst.BaseExpression) -> Optional[str]:
    """A comparable identity for a divisor: a name, a dotted name or an integer."""
    if isinstance(node, cst.Integer):
        return node.value
    return dotted_name(node)


class _GuardedStepFinder(cst.CSTVisitor):
    """Finds ``if (i + 1) % N == 0: optimizer.step()`` and reports the ``N``.

    That modulo guard *is* gradient accumulation: it is what makes several backward passes
    share one optimizer step. Without a guard there is one step per backward and nothing to
    scale, which is why the rule keys off it rather than off a variable name.
    """

    def __init__(self) -> None:
        self.divisors: List[cst.BaseExpression] = []

    def visit_If(self, node: cst.If) -> bool:
        divisor = self._modulo_divisor(node.test)
        if divisor is not None and _applies_gradients(node.body):
            self.divisors.append(divisor)
        return True

    def _modulo_divisor(self, test: cst.BaseExpression) -> Optional[cst.BaseExpression]:
        found: List[cst.BaseExpression] = []

        class _Modulo(cst.CSTVisitor):
            def visit_BinaryOperation(self, node: cst.BinaryOperation) -> bool:
                if isinstance(node.operator, cst.Modulo):
                    found.append(node.right)
                return True

        test.visit(_Modulo())
        return found[0] if found else None


class _StepFinder(cst.CSTVisitor):
    def __init__(self) -> None:
        self.found = False

    def visit_Call(self, node: cst.Call) -> bool:
        func = node.func
        if not isinstance(func, cst.Attribute) or func.attr.value != "step":
            return True
        receiver = (dotted_name(func.value) or "").lower().rsplit(".", 1)[-1]
        if "sched" in receiver:
            return True
        if any(hint in receiver for hint in OPTIMIZER_HINTS):
            self.found = True
        return True


def _applies_gradients(tree: cst.CSTNode) -> bool:
    finder = _StepFinder()
    tree.visit(finder)
    return finder.found


#: Calls that rescale the *gradients* before ``step()`` instead of dividing the loss.
#: torchtune weights each micro-batch loss by its token count and then applies
#: ``training.scale_grads_(params, 1.0 / num_tokens)``, which is a token-mean across
#: micro-batches of unequal length — a better normalisation than dividing by the step
#: count, not a missing one.
GRADIENT_RESCALE = "scale_grad"


def _mentions_rescale(tree: cst.CSTNode) -> bool:
    """Does any name in this expression refer to a gradient rescaler?"""

    class _Probe(cst.CSTVisitor):
        def __init__(self) -> None:
            self.found = False

        def visit_Name(self, node: cst.Name) -> bool:
            if GRADIENT_RESCALE in node.value:
                self.found = True
            return True

    probe = _Probe()
    tree.visit(probe)
    return probe.found


class _RescaleAliasFinder(cst.CSTVisitor):
    """Names bound to a gradient rescaler, so a call through one still counts.

    torchtune's distributed recipe does ``self._grad_scaler = training.scale_grads_`` and
    then calls ``self._grad_scaler(...)``, optionally wrapped in ``torch.compile(...)``.
    The call site's own name carries no evidence, which is the same shape as TG005 reading
    an attribute's name instead of the class it was bound to.
    """

    def __init__(self) -> None:
        self.aliases: set = set()

    def visit_Assign(self, node: cst.Assign) -> bool:
        if _mentions_rescale(node.value):
            for target in node.targets:
                name = dotted_name(target.target)
                if name:
                    self.aliases.add(name)
        return True


class _GradientRescaleFinder(cst.CSTVisitor):
    """Did anything rescale the gradients between the backwards and the step?"""

    def __init__(self, aliases: set) -> None:
        self.aliases = aliases
        self.found = False

    def visit_Call(self, node: cst.Call) -> bool:
        if GRADIENT_RESCALE in (final_attr(node.func) or ""):
            self.found = True
        elif (dotted_name(node.func) or "") in self.aliases:
            self.found = True
        return True


class _DivisionFinder(cst.CSTVisitor):
    """Any division or in-place division by one of the given divisors."""

    def __init__(self, divisors: set) -> None:
        self.divisors = divisors
        self.found = False

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> bool:
        if isinstance(node.operator, (cst.Divide, cst.FloorDivide)):
            if _key(node.right) in self.divisors:
                self.found = True
        return True

    def visit_AugAssign(self, node: cst.AugAssign) -> bool:
        if isinstance(node.operator, cst.DivideAssign) and _key(node.value) in self.divisors:
            self.found = True
        return True


@register
class AccumulationWithoutScaling(Rule):
    code = "TG014"
    name = "unscaled-gradient-accumulation"
    summary = "Gradient accumulation without dividing the loss"
    severity = Severity.ERROR
    category = Category.CONVERGENCE_BUG
    explanation = """
Gradient accumulation runs several backward passes before one optimizer step, so the
gradients of those micro-batches **sum** into ``.grad``. A real batch of that size would
have *averaged* them, so unless the loss is divided by the accumulation count, the gradient
is N times too large — arithmetically identical to multiplying the learning rate by N.

Nothing crashes. With N=4 and a tuned learning rate the run behaves as if the rate were 4x
too high: it may diverge, or plateau somewhere worse, and the loss curve looks like a bad
hyperparameter rather than a bug. Doubling the accumulation count to fit a bigger effective
batch then makes it worse, which sends people tuning the learning rate instead of fixing
the scaling.

Divide before the backward pass::

    (loss / accumulation_steps).backward()

Reduction matters too: this assumes the default ``reduction="mean"``. With
``reduction="sum"`` the arithmetic is different and the division may not be what you want.

``accelerator.accumulate(...)``, the Hugging Face ``Trainer``, Lightning and DeepSpeed all
scale internally — dividing again there would shrink the gradient by N instead. The rule
stays silent when it can see one of those in the file.
""".strip()

    _aliases: Optional[set] = None

    def _rescale_aliases(self) -> set:
        """Computed once per file, on the first backward this rule reaches."""
        if self._aliases is None:
            finder = _RescaleAliasFinder()
            self.ctx.module.visit(finder)
            self._aliases = finder.aliases
        return self._aliases

    def visit_Call(self, node: cst.Call) -> bool:
        func = node.func
        if not isinstance(func, cst.Attribute) or func.attr.value != "backward":
            return True

        if self.ctx.is_lightning:
            return True
        owner = (dotted_name(func.value) or "").lower()
        if any(marker in owner for marker in MANAGED_ACCUMULATION):
            return True

        loop = self.innermost_loop
        if loop is None:
            return True

        # A framework that owns the scaling anywhere in the file is enough to stay quiet;
        # a false positive here tells someone to introduce a real bug.
        lowered = self.ctx.source.lower()
        if any(marker in lowered for marker in MANAGED_ACCUMULATION):
            return True

        guards = _GuardedStepFinder()
        loop.node.visit(guards)
        if not guards.divisors:
            return True  # one step per backward: nothing accumulates, nothing to scale

        divisors = {key for key in (_key(d) for d in guards.divisors) if key}
        if not divisors:
            return True
        # ``% 1`` is not accumulation, and dividing by it would be a no-op anyway.
        divisors.discard("1")
        if not divisors:
            return True

        division = _DivisionFinder(divisors)
        loop.node.visit(division)
        if division.found:
            return True

        # Dividing the loss is the common compensation, not the only one. Scaling the
        # gradients before `step()` achieves the same thing, and telling those callers to
        # divide as well would shrink the gradient by the accumulation factor.
        rescale = _GradientRescaleFinder(self._rescale_aliases())
        loop.node.visit(rescale)
        if rescale.found:
            return True

        divisor = sorted(divisors)[0]
        self.report(
            func.value,
            f"Gradients accumulate over `{divisor}` micro-batches before `step()`, but the "
            f"loss is never divided by it — the summed gradient is `{divisor}`x too large, "
            f"which is the same as scaling the learning rate by `{divisor}`.",
            hint=f"Use `(loss / {divisor}).backward()`, assuming the default "
                 f'`reduction="mean"`.',
            fix_build=_scale_loss(divisor),
            fix_description=f"divide the loss by {divisor}",
        )
        return True


def _scale_loss(divisor: str):
    """Rewrite ``loss.backward()`` as ``(loss / N).backward()``.

    The reported node is the receiver of ``.backward()``, so replacing it with a
    parenthesised division scales only what autograd sees. Anything that logs or
    accumulates ``loss`` afterwards keeps reporting the same value it did before, which is
    why this is preferred over reassigning ``loss``.
    """

    def build(updated: cst.CSTNode) -> cst.CSTNode:
        assert isinstance(updated, cst.BaseExpression)
        return cst.BinaryOperation(
            left=updated,
            operator=cst.Divide(),
            right=cst.parse_expression(divisor),
            lpar=[cst.LeftParen()],
            rpar=[cst.RightParen()],
        )

    return build
