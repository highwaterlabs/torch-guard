"""TG006 - binary cross-entropy paired with the wrong activation."""

from __future__ import annotations

from typing import Optional

import libcst as cst

from ..analysis.helpers import dotted_name, final_attr, keyword_arg, positional_args
from ..diagnostics import Category, Severity
from .base import Rule, register

SIGMOID_CALLS = {"sigmoid"}

#: ``BCELoss`` expects probabilities; ``BCEWithLogitsLoss`` expects raw logits and applies
#: the sigmoid itself, inside a log-sum-exp that does not overflow.
BCE_PLAIN = {"BCELoss", "binary_cross_entropy"}
BCE_WITH_LOGITS = {"BCEWithLogitsLoss", "binary_cross_entropy_with_logits"}


@register
class BinaryCrossEntropyActivation(Rule):
    code = "TG006"
    name = "bce-activation-mismatch"
    summary = "Binary cross-entropy paired with the wrong activation"
    severity = Severity.ERROR
    category = Category.CONVERGENCE_BUG
    explanation = """
The two binary cross-entropy losses expect different inputs, and pairing them wrongly is
silent in three of the four combinations.

``nn.BCEWithLogitsLoss`` / ``F.binary_cross_entropy_with_logits`` want **raw logits** and
apply the sigmoid internally. Feeding them ``sigmoid(x)`` applies it twice: the effective
input is squashed into a narrow band, gradients shrink, and training stalls at a poor
optimum without ever erroring.

``nn.BCELoss`` / ``F.binary_cross_entropy`` want **probabilities in [0, 1]**. Feeding raw
logits is worse than wrong: the loss takes ``log`` of a negative number, so it produces
``nan`` immediately, and any negative logit makes the whole batch ``nan``.

Even correctly paired, ``sigmoid`` followed by ``BCELoss`` is the numerically fragile
combination. ``BCEWithLogitsLoss`` folds the sigmoid into a log-sum-exp that cannot
overflow; the separate version computes ``log(sigmoid(x))``, which underflows to ``-inf``
for confident predictions. The PyTorch docs recommend the fused version for exactly this
reason.

Prefer ``BCEWithLogitsLoss`` on raw logits, and keep ``sigmoid`` for reporting
probabilities at inference time.
""".strip()

    def visit_Call(self, node: cst.Call) -> bool:
        self._check_layer(node)

        kind = self._loss_kind(node)
        if kind is None:
            return True

        arg = self._first_input(node)
        if arg is None:
            return True

        inline_sigmoid = self._sigmoid_of(arg)
        name = dotted_name(arg)
        via_variable = bool(name and name in self.ctx.sigmoid_vars)

        if kind == "logits" and (inline_sigmoid or via_variable):
            self._report_double_sigmoid(arg, inline_sigmoid, name)
        elif kind == "plain" and (inline_sigmoid or via_variable):
            self._report_unstable_pairing(arg, inline_sigmoid, name)
        elif kind == "plain":
            self._report_possible_logits(arg, name)
        return True

    # ------------------------------------------------------------------ cases

    def _report_double_sigmoid(self, arg, inline: Optional[str], name: str) -> None:
        subject = "`sigmoid(...)`" if inline else f"`{name}` (output of `sigmoid`)"
        self.report(
            arg,
            f"{subject} is passed to `BCEWithLogitsLoss`, which applies `sigmoid` itself - "
            f"the activation is applied twice, shrinking gradients and stalling training.",
            hint="Pass the raw logits directly; the loss applies the sigmoid itself.",
            fix_build=_unwrap_sigmoid if inline else None,
            fix_description="remove redundant sigmoid" if inline else None,
        )

    def _report_unstable_pairing(self, arg, inline: Optional[str], name: str) -> None:
        subject = "`sigmoid(...)`" if inline else f"`{name}` (output of `sigmoid`)"
        self.report(
            arg,
            f"{subject} with `BCELoss` is correct but numerically fragile: "
            f"`log(sigmoid(x))` underflows to `-inf` for confident predictions.",
            hint="Use `BCEWithLogitsLoss` on the raw logits instead; it fuses the sigmoid "
                 "into a log-sum-exp that cannot overflow.",
            severity=Severity.WARNING,
        )

    def _report_possible_logits(self, arg, name: str) -> None:
        """``BCELoss`` fed something that carries a graph, with no sigmoid in the file.

        The absence of *any* sigmoid is the evidence. A file that computes probabilities
        somewhere is not making this mistake; a file with none is feeding raw logits to a
        loss that takes their logarithm, which is ``nan`` on the first negative value.
        """
        if self.ctx.has_sigmoid:
            return
        if not name or not self.prov.is_grad_name(name, self.scope_path):
            return
        self.report(
            arg,
            f"`{name}` is passed to `BCELoss`, which expects probabilities in [0, 1], and "
            f"nothing in this file applies a `sigmoid` - raw logits produce `nan` as soon "
            f"as one is negative.",
            hint="Use `BCEWithLogitsLoss`, which takes the logits directly and is "
                 "numerically stable.",
        )

    def _check_layer(self, node: cst.Call) -> None:
        """A model whose ``nn.Sequential`` *ends* in ``nn.Sigmoid()``, with the fused loss.

        Deliberately narrow. An earlier version flagged any ``nn.Sigmoid()`` construction
        in a file that mentioned ``BCEWithLogitsLoss`` anywhere, which produced three false
        positives in ``torch/testing/_internal/common_nn.py``: a bare
        ``sigmoid = nn.Sigmoid()`` local, used to build a *reference* implementation, is
        not a model ending in a sigmoid. Final position in a ``Sequential`` is evidence;
        merely constructing the layer is not.
        """
        if not self.ctx.uses_bce_with_logits:
            return
        if final_attr(node.func) != "Sequential":
            return
        dotted = dotted_name(node.func) or ""
        if not (dotted.startswith(("nn.", "torch.nn.")) or dotted == "Sequential"):
            return

        args = positional_args(node)
        if not args:
            return
        last = args[-1].value
        if not isinstance(last, cst.Call) or final_attr(last.func) != "Sigmoid":
            return

        self.report(
            last,
            "Model ends with `nn.Sigmoid()` but `BCEWithLogitsLoss` in this file already "
            "applies a sigmoid; the activation is applied twice.",
            hint="Drop the final activation layer and return raw logits from `forward`.",
            severity=Severity.WARNING,
        )

    # ------------------------------------------------------------------ helpers

    def _loss_kind(self, node: cst.Call) -> Optional[str]:
        """'logits' for the fused loss, 'plain' for the probability one."""
        leaf = final_attr(node.func)
        if leaf in BCE_WITH_LOGITS:
            return "logits"
        if leaf in BCE_PLAIN:
            return "plain"

        cls = self.prov.criterion_class(dotted_name(node.func), self.scope_path)
        if cls in BCE_WITH_LOGITS:
            return "logits"
        if cls in BCE_PLAIN:
            return "plain"
        return None

    def _first_input(self, node: cst.Call) -> Optional[cst.BaseExpression]:
        args = positional_args(node)
        if args:
            return args[0].value
        kw = keyword_arg(node, "input")
        return kw.value if kw is not None else None

    def _sigmoid_of(self, node: cst.BaseExpression) -> Optional[str]:
        """``torch.sigmoid(x)``, ``F.sigmoid(x)`` or ``x.sigmoid()``."""
        if isinstance(node, cst.Call) and final_attr(node.func) in SIGMOID_CALLS:
            return "sigmoid"
        return None


def _unwrap_sigmoid(updated: cst.CSTNode) -> cst.CSTNode:
    """Replace ``sigmoid(logits)`` with ``logits``.

    ``x.sigmoid()`` takes no arguments, so the logits are the receiver rather than the
    first argument; unwrapping the wrong one would silently change what is being trained.
    """
    assert isinstance(updated, cst.Call)
    args = positional_args(updated)
    if args:
        return args[0].value
    if isinstance(updated.func, cst.Attribute):
        return updated.func.value
    kw = keyword_arg(updated, "input")
    return kw.value if kw is not None else updated
