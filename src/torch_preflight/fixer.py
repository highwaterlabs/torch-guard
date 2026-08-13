"""Autofix application.

Rules record fixes as ``(node, builder)`` pairs during their read-only pass. This module
replays them in a single LibCST transform, which is why fixes preserve formatting and
comments: only the targeted nodes change, everything else round-trips byte for byte.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import libcst as cst

from .diagnostics import Diagnostic


class _FixApplier(cst.CSTTransformer):
    def __init__(self, fixes: Dict[int, Diagnostic]) -> None:
        super().__init__()
        self.fixes = fixes
        self.applied: List[Diagnostic] = []

    def on_leave(self, original_node, updated_node):  # type: ignore[override]
        result = super().on_leave(original_node, updated_node)
        diagnostic = self.fixes.get(id(original_node))
        if diagnostic is None or not isinstance(result, cst.CSTNode):
            return result
        assert diagnostic.fix is not None
        replacement = diagnostic.fix.build(result)
        self.applied.append(diagnostic)
        return replacement


def apply_fixes(
    module: cst.Module, diagnostics: Iterable[Diagnostic]
) -> Tuple[str, List[Diagnostic]]:
    """Return ``(new_source, applied)`` after applying every fixable diagnostic.

    Nodes are matched by identity, so ``module`` must be the exact tree the rules ran
    against (see ``build_context``'s ``unsafe_skip_copy``).
    """
    fixes: Dict[int, Diagnostic] = {}
    for diagnostic in diagnostics:
        if diagnostic.fix is None:
            continue
        # If two rules target the same node, first one wins — deterministic because
        # diagnostics are sorted by code before they reach here.
        fixes.setdefault(diagnostic.fix.target_id, diagnostic)

    if not fixes:
        return module.code, []

    applier = _FixApplier(fixes)
    new_module = module.visit(applier)
    return new_module.code, applier.applied
