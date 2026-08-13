"""Diagnostic data model shared by the rule engine, reporters and the fixer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import libcst as cst


class Severity(str, Enum):
    """How loudly a finding should be reported. Drives exit codes and CI annotations."""

    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {Severity.NOTE: 0, Severity.WARNING: 1, Severity.ERROR: 2}


class Category(str, Enum):
    """What kind of damage the finding causes. This is the axis ML engineers care about."""

    CRITICAL_OOM = "CRITICAL_OOM"
    CONVERGENCE_BUG = "CONVERGENCE_BUG"
    PERFORMANCE_WARN = "PERFORMANCE_WARN"


# A fix builder receives the *updated* node (post child-transformation) and returns
# its replacement. Keeping it lazy lets rules record fixes during a read-only pass.
FixBuilder = Callable[[cst.CSTNode], cst.CSTNode]


@dataclass(frozen=True)
class Fix:
    """An autofix: replace ``target`` with ``build(updated_target)``.

    The node reference is held (not just its ``id``) so the tree stays alive for as long
    as the fix does — the fixer matches nodes by identity.
    """

    target: cst.CSTNode
    build: FixBuilder
    description: str

    @property
    def target_id(self) -> int:
        return id(self.target)


@dataclass
class Diagnostic:
    code: str
    message: str
    severity: Severity
    category: Category
    line: int
    column: int
    path: str = "<unknown>"
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    hint: Optional[str] = None
    fix: Optional[Fix] = field(default=None, repr=False)
    #: The node this finding points at. Positions are resolved after traversal, only for
    #: files that actually produced findings — see ``engine.check_source``.
    node: Optional[cst.CSTNode] = field(default=None, repr=False, compare=False)
    #: Set when a fix was stripped to make the diagnostic picklable (worker processes).
    #: ``--fix`` always runs single-process, so the real fix is never lost.
    fix_label: Optional[str] = None

    def resolve_position(self, positions) -> None:
        """Fill in line/column from a resolved PositionProvider mapping."""
        if self.node is None:
            return
        span = positions[self.node]
        self.line = span.start.line
        self.column = span.start.column + 1
        self.end_line = span.end.line
        self.end_column = span.end.column + 1
        self.node = None

    @property
    def fixable(self) -> bool:
        return self.fix is not None or self.fix_label is not None

    @property
    def fix_summary(self) -> Optional[str]:
        if self.fix is not None:
            return self.fix.description
        return self.fix_label

    @property
    def location(self) -> str:
        return f"{self.path}:{self.line}:{self.column}"

    def sort_key(self) -> tuple:
        return (self.path, self.line, self.column, self.code)
