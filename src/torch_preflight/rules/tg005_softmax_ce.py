"""TG005 - activation applied before a loss that applies it internally."""

from __future__ import annotations

from typing import Optional

import libcst as cst

from ..analysis.helpers import dotted_name, final_attr, keyword_arg, positional_args
from ..diagnostics import Category, Severity
from .base import Rule, register

SOFTMAX_CALLS = {"softmax", "log_softmax"}
SOFTMAX_LAYERS = {"Softmax", "LogSoftmax"}


@register
class SoftmaxBeforeCrossEntropy(Rule):
    code = "TG005"
    name = "softmax-before-cross-entropy"
    summary = "Softmax applied before a loss that expects raw logits"
    severity = Severity.ERROR
    category = Category.CONVERGENCE_BUG
    explanation = """
``nn.CrossEntropyLoss`` / ``F.cross_entropy`` apply ``log_softmax`` internally and expect
**raw logits**. Feeding them softmax probabilities applies the normalisation twice: the
effective logits are squashed toward uniform, gradients shrink sharply, and the model
trains far slower or plateaus at a poor accuracy. Nothing crashes and the loss still
decreases, which is why this survives code review.

``nn.NLLLoss`` is the mirror image - it expects ``log_softmax`` output, so feeding it plain
``softmax`` probabilities is also wrong.

Pass logits straight to ``CrossEntropyLoss``, and keep ``softmax`` for inference-time
probability reporting only.
""".strip()

    # -------------------------------------------------- case 1/2: at the call site

    def visit_Call(self, node: cst.Call) -> bool:
        self._check_layer(node)

        kind = self._loss_kind(node)
        if kind is None:
            return True

        args = positional_args(node)
        if args:
            arg = args[0].value
        else:
            kw = keyword_arg(node, "input")
            if kw is None:
                return True
            arg = kw.value

        # Directly wrapped: ``criterion(F.softmax(logits, 1), y)``
        activation = self._activation_of(arg)
        if activation is not None and self._is_mismatch(kind, activation):
            self.report(
                arg,
                self._message(kind, activation, inline=True),
                hint="Pass the raw logits directly; the loss applies the activation itself.",
                fix_build=_unwrap_activation,
                fix_description=f"remove redundant {activation}",
            )
            return True

        # Indirect: ``probs = F.softmax(logits, 1)`` ... ``criterion(probs, y)``
        name = dotted_name(arg)
        if name and name in self.ctx.softmax_vars:
            indirect = self.ctx.softmax_vars[name]
            if self._is_mismatch(kind, indirect):
                self.report(
                    arg,
                    self._message(kind, indirect, inline=False, var=name),
                    hint=f"Pass the logits that `{name}` was computed from instead.",
                )
        return True

    # ------------------------------------------- case 3: an activation in the model

    def _check_layer(self, node: cst.Call) -> None:
        """A model whose ``nn.Sequential`` *ends* in a softmax layer, with a fused loss.

        Deliberately narrow, for the same reason TG006's sigmoid check is. An earlier
        version flagged any ``nn.Softmax(...)`` construction in a file that mentioned
        ``NLLLoss`` or ``CrossEntropyLoss`` anywhere, and fired on
        ``pytorch/examples/gat/main.py``, where ``self.softmax = nn.Softmax(dim=1)``
        normalises **attention coefficients** over neighbours and the model correctly ends
        in ``F.log_softmax``. Attention softmax appears in every transformer and GNN, so
        constructing the layer cannot be the evidence — final position in a ``Sequential``
        can be.
        """
        if final_attr(node.func) != "Sequential":
            return
        sequential = dotted_name(node.func) or ""
        if not (sequential.startswith(("nn.", "torch.nn.")) or sequential == "Sequential"):
            return

        args = positional_args(node)
        if not args:
            return
        node = args[-1].value
        if not isinstance(node, cst.Call):
            return
        leaf = final_attr(node.func)
        if leaf not in SOFTMAX_LAYERS:
            return

        if leaf == "Softmax" and self.ctx.uses_cross_entropy:
            detail = "`nn.CrossEntropyLoss` in this file already applies `log_softmax`"
        elif leaf == "LogSoftmax" and self.ctx.uses_cross_entropy and not self.ctx.uses_nll_loss:
            detail = "`nn.CrossEntropyLoss` in this file already applies `log_softmax`"
        elif leaf == "Softmax" and self.ctx.uses_nll_loss:
            detail = "`nn.NLLLoss` in this file expects log-probabilities, not probabilities"
        else:
            return

        self.report(
            node,
            f"Model ends with `nn.{leaf}(...)` but {detail}; the activation is applied "
            f"twice and gradients are badly scaled.",
            hint="Drop the final activation layer and return raw logits from `forward`.",
            severity=Severity.WARNING,
        )

    # ------------------------------------------------------------------ helpers

    def _loss_kind(self, node: cst.Call) -> Optional[str]:
        """Classify the callee as a cross-entropy ('ce') or NLL ('nll') loss."""
        leaf = final_attr(node.func)
        if leaf == "cross_entropy":
            return "ce"
        if leaf == "nll_loss":
            return "nll"

        dotted = dotted_name(node.func)
        cls = self.prov.criterion_class(dotted, self.scope_path)
        if cls == "CrossEntropyLoss":
            return "ce"
        if cls == "NLLLoss":
            return "nll"
        return None

    def _activation_of(self, node: cst.BaseExpression) -> Optional[str]:
        if isinstance(node, cst.Call):
            leaf = final_attr(node.func)
            if leaf in SOFTMAX_CALLS:
                return leaf
        return None

    def _is_mismatch(self, kind: str, activation: str) -> bool:
        if kind == "ce":
            return True  # CE wants logits; both softmax and log_softmax are wrong
        return activation == "softmax"  # NLLLoss wants log_softmax, not softmax

    def _message(self, kind: str, activation: str, *, inline: bool, var: str = "") -> str:
        loss = "`CrossEntropyLoss`" if kind == "ce" else "`NLLLoss`"
        subject = f"`{activation}(...)`" if inline else f"`{var}` (output of `{activation}`)"
        if kind == "ce":
            return (
                f"{subject} is passed to {loss}, which applies `log_softmax` itself - the "
                f"activation is applied twice, shrinking gradients and slowing convergence."
            )
        return (
            f"{subject} is passed to {loss}, which expects log-probabilities; feeding it "
            f"plain probabilities computes the wrong loss."
        )


def _unwrap_activation(updated: cst.CSTNode) -> cst.CSTNode:
    """Replace ``softmax(logits, dim=1)`` with ``logits``."""
    assert isinstance(updated, cst.Call)
    args = positional_args(updated)
    if args:
        return args[0].value
    kw = keyword_arg(updated, "input")
    return kw.value if kw is not None else updated
