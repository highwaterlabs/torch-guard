"""Rule base class, dispatcher and registry.

Rules are **not** visitors. A single :class:`RuleDispatcher` walks the tree once,
maintains scope/loop/no-grad state once, and forwards each node to whichever rules asked
for that node type. Rules still look like visitors — they define ``visit_Call``,
``leave_FunctionDef`` and so on, and read ``self.in_loop`` — but the state they read
belongs to the dispatcher.

This matters: with six rules each running its own traversal, 78% of analysis time went to
walking the same tree six times. Parsing was 7%.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Type

import libcst as cst

from ..analysis.context import FileContext
from ..analysis.scope import LoopFrame, Scope, ScopePath, ScopeTrackingVisitor
from ..diagnostics import Category, Diagnostic, Fix, FixBuilder, Severity


class Rule:
    """A single check.

    Subclasses set the class-level metadata below and define ordinary
    ``visit_<Node>`` / ``leave_<Node>`` methods, calling :meth:`report` on a finding.
    """

    code: str = ""
    name: str = ""
    summary: str = ""
    severity: Severity = Severity.WARNING
    category: Category = Category.PERFORMANCE_WARN
    #: Longer explanation shown by ``torch-preflight explain <code>``.
    explanation: str = ""

    def __init__(self, ctx: FileContext) -> None:
        self.ctx = ctx
        self.diagnostics: List[Diagnostic] = []
        #: Set by the engine; only rules needing project settings read it.
        self.cfg = None
        #: The dispatcher, which owns the traversal state. Set before traversal.
        self._state: Optional["RuleDispatcher"] = None

    # ------------------------------------------------------------------ scheduling

    @classmethod
    def should_run(cls, ctx: FileContext, cfg) -> bool:
        """Cheap pre-check: can this rule possibly fire on this file?

        Returning False skips the rule entirely. Since traversal is shared, this only
        saves the rule's own per-node work — but for a rule like TG010, which does a
        second full extraction pass, it saves a great deal.
        """
        return True

    # ------------------------------------------------- state, owned by the dispatcher

    @property
    def scopes(self) -> List[Scope]:
        return self._state.scopes

    @property
    def loops(self) -> List[LoopFrame]:
        return self._state.loops

    @property
    def scope_path(self) -> ScopePath:
        return self._state.scope_path

    @property
    def in_no_grad(self) -> bool:
        return self._state.in_no_grad

    @property
    def in_loop(self) -> bool:
        return self._state.in_loop

    @property
    def innermost_loop(self) -> Optional[LoopFrame]:
        return self._state.innermost_loop

    @property
    def current_function(self) -> Optional[Scope]:
        return self._state.current_function

    @property
    def current_class(self) -> Optional[Scope]:
        return self._state.current_class

    def bound_in_innermost_loop(self, name: str) -> bool:
        return self._state.bound_in_innermost_loop(name)

    def bound_in_any_loop(self, name: str) -> bool:
        return self._state.bound_in_any_loop(name)

    # ------------------------------------------------------------------ helpers

    @property
    def prov(self):
        return self.ctx.provenance

    def is_grad(self, node: cst.BaseExpression) -> bool:
        loops = [id(frame.node) for frame in self.loops]
        return self.prov.is_grad_bearing(node, self.scope_path, loops)

    def report(
        self,
        node: cst.CSTNode,
        message: str,
        *,
        hint: Optional[str] = None,
        fix_node: Optional[cst.CSTNode] = None,
        fix_build: Optional[FixBuilder] = None,
        fix_description: str = "",
        severity: Optional[Severity] = None,
        category: Optional[Category] = None,
    ) -> None:
        fix = None
        if fix_build is not None:
            fix = Fix(fix_node if fix_node is not None else node, fix_build, fix_description)

        self.diagnostics.append(
            Diagnostic(
                code=self.code,
                message=message,
                severity=severity or self.severity,
                category=category or self.category,
                # Filled in after traversal, only if this file produced findings.
                line=0,
                column=0,
                path=self.ctx.path,
                hint=hint,
                fix=fix,
                node=node,
            )
        )

    def report_at_line(
        self,
        line: int,
        message: str,
        *,
        hint: Optional[str] = None,
        severity: Optional[Severity] = None,
    ) -> None:
        """Report against a line number, for findings not anchored to one node."""
        self.diagnostics.append(
            Diagnostic(
                code=self.code,
                message=message,
                severity=severity or self.severity,
                category=self.category,
                line=line,
                column=1,
                end_line=line,
                path=self.ctx.path,
                hint=hint,
            )
        )


class RuleDispatcher(ScopeTrackingVisitor):
    """Walks the tree once and fans each node out to the rules that want it.

    Deliberately declares no metadata dependencies. Resolving ``PositionProvider`` costs
    more than parsing the file, and the overwhelming majority of files produce no
    findings at all, so positions are resolved afterwards and only when needed.
    """

    def __init__(self, rules: Sequence[Rule]) -> None:
        super().__init__()
        self.rules = list(rules)
        # node class -> handlers, resolved once so dispatch is a single dict lookup.
        self._on_visit: Dict[type, List[Callable]] = {}
        self._on_leave: Dict[type, List[Callable]] = {}

        for rule in self.rules:
            rule._state = self
            for attribute in dir(rule.__class__):
                if attribute.startswith("visit_"):
                    table, suffix = self._on_visit, attribute[len("visit_"):]
                elif attribute.startswith("leave_"):
                    table, suffix = self._on_leave, attribute[len("leave_"):]
                else:
                    continue
                node_type = getattr(cst, suffix, None)
                if isinstance(node_type, type):
                    table.setdefault(node_type, []).append(getattr(rule, attribute))

    def on_visit(self, node: cst.CSTNode) -> bool:
        self._push(node)
        handlers = self._on_visit.get(node.__class__)
        if handlers is not None:
            for handler in handlers:
                handler(node)
        # Always descend. No built-in rule prunes the tree, and one rule opting out must
        # not hide nodes from the others.
        return True

    def on_leave(self, original_node: cst.CSTNode) -> None:
        handlers = self._on_leave.get(original_node.__class__)
        if handlers is not None:
            for handler in handlers:
                handler(original_node)
        self._pop(original_node)


RULES: Dict[str, Type[Rule]] = {}


def register(cls: Type[Rule]) -> Type[Rule]:
    """Class decorator adding a rule to the global registry."""
    if not cls.code:
        raise ValueError(f"{cls.__name__} must define a code")
    if cls.code in RULES:
        raise ValueError(f"duplicate rule code {cls.code}")
    RULES[cls.code] = cls
    return cls


def all_rules() -> List[Type[Rule]]:
    return [RULES[code] for code in sorted(RULES)]
