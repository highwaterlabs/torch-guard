"""TG004 - DataLoader settings that starve the GPU."""

from __future__ import annotations

import libcst as cst

from ..analysis.helpers import (
    final_attr,
    is_literal_false,
    is_literal_zero,
    keyword_arg,
)
from ..diagnostics import Category, Severity
from .base import Rule, register


@register
class DataLoaderStarvation(Rule):
    code = "TG004"
    name = "dataloader-starvation"
    summary = "DataLoader configured to stall a CUDA device"
    severity = Severity.WARNING
    category = Category.PERFORMANCE_WARN
    explanation = """
With ``num_workers=0`` the DataLoader runs in the training process, so the GPU sits idle
during every decode/augment step - the classic sawtooth utilisation graph. On a multi-GPU
DDP node this multiplies: every rank stalls on the same single-threaded CPU pipeline.

``pin_memory=True`` allocates page-locked staging buffers, which is what makes
``.to(device, non_blocking=True)` transfers overlap with compute instead of blocking.

Rules of thumb: ``num_workers`` = 4 x number of GPUs (bounded by CPU cores), and
``pin_memory=True`` whenever batches end up on CUDA.
""".strip()

    def visit_Call(self, node: cst.Call) -> bool:
        if final_attr(node.func) != "DataLoader":
            return True
        # Only meaningful when the batches are headed for a GPU.
        if not self.ctx.uses_cuda:
            return True

        workers = keyword_arg(node, "num_workers")
        if workers is None:
            self.report(
                node,
                "DataLoader does not set `num_workers`, so it defaults to 0 and loads "
                "batches in the training process while the GPU idles.",
                hint="Set `num_workers` to roughly 4x your GPU count, capped by CPU cores.",
            )
        elif is_literal_zero(workers.value):
            self.report(
                node,
                "DataLoader uses `num_workers=0`: data loading blocks the training loop "
                "and the GPU stalls between batches.",
                hint="Set `num_workers` to roughly 4x your GPU count, capped by CPU cores.",
            )

        pin = keyword_arg(node, "pin_memory")
        if pin is None or is_literal_false(pin.value):
            self.report(
                node,
                "DataLoader does not enable `pin_memory`, so host-to-device copies cannot "
                "overlap with compute.",
                hint="Pass `pin_memory=True` and use `.to(device, non_blocking=True)`.",
                fix_build=_set_pin_memory if pin is None else None,
                fix_description="add pin_memory=True",
            )
        return True


def _set_pin_memory(updated: cst.CSTNode) -> cst.CSTNode:
    """Append ``pin_memory=True``. Relies on ``MaybeSentinel`` to render the comma."""
    assert isinstance(updated, cst.Call)
    new_arg = cst.Arg(
        keyword=cst.Name("pin_memory"),
        value=cst.Name("True"),
        equal=cst.AssignEqual(
            whitespace_before=cst.SimpleWhitespace(""),
            whitespace_after=cst.SimpleWhitespace(""),
        ),
    )
    return updated.with_changes(args=[*updated.args, new_arg])
