"""Scope, loop and autograd-context tracking.

This is the state layer that separates torch-preflight from a plain AST linter: rules
can ask "am I inside a loop?", "is autograd disabled here?" and "which enclosing
function am I in?" without re-walking the tree.

Subclasses of :class:`ScopeTrackingVisitor` may define ordinary ``visit_<Node>`` /
``leave_<Node>`` methods. State is pushed in ``on_visit`` before dispatch and popped
in ``on_leave`` after dispatch, so no cooperative ``super()`` calls are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import libcst as cst

from .helpers import NO_GRAD_NAMES, dotted_name, final_attr, is_literal_false

ScopePath = Tuple[str, ...]

#: Node types that change scope, loop or autograd state. Everything else is skipped with
#: a single hash lookup — most nodes in a tree are Name, Attribute, Arg and whitespace.
_STATEFUL_TYPES = frozenset({
    cst.FunctionDef, cst.ClassDef, cst.Lambda,
    cst.ListComp, cst.SetComp, cst.DictComp, cst.GeneratorExp,
    cst.For, cst.While, cst.With,
    cst.Assign, cst.AnnAssign, cst.NamedExpr,
})


@dataclass
class Scope:
    kind: str  # "module" | "class" | "function" | "comprehension"
    name: str
    node: Optional[cst.CSTNode] = None
    assigned: Set[str] = field(default_factory=set)


@dataclass
class LoopFrame:
    node: cst.CSTNode
    #: Names bound inside this loop body (targets + assignments). A container
    #: created inside the loop cannot accumulate across iterations.
    assigned: Set[str] = field(default_factory=set)
    #: Dotted name of the iterable, e.g. ``val_loader`` in ``for b in val_loader``.
    iterable: Optional[str] = None


def _is_no_grad_expr(node: cst.BaseExpression) -> bool:
    """True for ``torch.no_grad()``, ``inference_mode()``, ``set_grad_enabled(False)``."""
    if isinstance(node, cst.Call):
        name = final_attr(node.func)
        if name in NO_GRAD_NAMES:
            return True
        if name == "set_grad_enabled":
            args = [a for a in node.args if a.keyword is None]
            return bool(args) and is_literal_false(args[0].value)
        return False
    # Bare ``@torch.no_grad`` (no parens) is legal as a decorator.
    return final_attr(node) in NO_GRAD_NAMES


def function_disables_grad(node: cst.FunctionDef) -> bool:
    """True if the function is decorated with ``@torch.no_grad``/``@torch.inference_mode``."""
    return any(_is_no_grad_expr(d.decorator) for d in node.decorators)


def with_disables_grad(node: cst.With) -> bool:
    return any(_is_no_grad_expr(item.item) for item in node.items)


class ScopeTrackingVisitor(cst.CSTVisitor):
    """Base visitor maintaining scope, loop and no-grad state."""

    def __init__(self) -> None:
        super().__init__()
        self.scopes: List[Scope] = [Scope("module", "<module>")]
        self.loops: List[LoopFrame] = []
        self._no_grad_depth = 0

    # ---------------------------------------------------------------- state API

    @property
    def scope_path(self) -> ScopePath:
        return tuple(s.name for s in self.scopes)

    @property
    def in_no_grad(self) -> bool:
        return self._no_grad_depth > 0

    @property
    def in_loop(self) -> bool:
        return bool(self.loops)

    @property
    def innermost_loop(self) -> Optional[LoopFrame]:
        return self.loops[-1] if self.loops else None

    @property
    def current_function(self) -> Optional[Scope]:
        for scope in reversed(self.scopes):
            if scope.kind == "function":
                return scope
        return None

    @property
    def current_class(self) -> Optional[Scope]:
        for scope in reversed(self.scopes):
            if scope.kind == "class":
                return scope
        return None

    def note_binding(self, name: str) -> None:
        """Record that ``name`` was bound in the current scope and innermost loop."""
        self.scopes[-1].assigned.add(name)
        if self.loops:
            self.loops[-1].assigned.add(name)

    def bound_in_innermost_loop(self, name: str) -> bool:
        return bool(self.loops) and name in self.loops[-1].assigned

    def bound_in_any_loop(self, name: str) -> bool:
        return any(name in frame.assigned for frame in self.loops)

    # ------------------------------------------------------------- CST plumbing

    def on_visit(self, node: cst.CSTNode) -> bool:
        self._push(node)
        return super().on_visit(node)

    def on_leave(self, original_node: cst.CSTNode) -> None:
        super().on_leave(original_node)
        self._pop(original_node)

    def _push(self, node: cst.CSTNode) -> None:
        if node.__class__ not in _STATEFUL_TYPES:
            return
        if isinstance(node, cst.FunctionDef):
            self.scopes.append(Scope("function", node.name.value, node))
            if function_disables_grad(node):
                self._no_grad_depth += 1
        elif isinstance(node, cst.ClassDef):
            self.scopes.append(Scope("class", node.name.value, node))
        elif isinstance(node, cst.Lambda):
            self.scopes.append(Scope("function", "<lambda>", node))
        elif isinstance(node, (cst.ListComp, cst.SetComp, cst.DictComp, cst.GeneratorExp)):
            self.scopes.append(Scope("comprehension", "<comp>", node))
        elif isinstance(node, (cst.For, cst.While)):
            frame = LoopFrame(node)
            if isinstance(node, cst.For):
                frame.iterable = dotted_name(node.iter)
                for name in target_names(node.target):
                    frame.assigned.add(name)
            self.loops.append(frame)
        elif isinstance(node, cst.With):
            if with_disables_grad(node):
                self._no_grad_depth += 1
        elif isinstance(node, (cst.Assign, cst.AnnAssign, cst.NamedExpr)):
            # ``AugAssign`` is deliberately excluded: ``total += loss`` mutates a name
            # that must already exist, so it does not make the name loop-local.
            for name in _assignment_target_names(node):
                self.note_binding(name)

    def _pop(self, node: cst.CSTNode) -> None:
        if node.__class__ not in _STATEFUL_TYPES:
            return
        if isinstance(node, cst.FunctionDef):
            self.scopes.pop()
            if function_disables_grad(node):
                self._no_grad_depth -= 1
        elif isinstance(node, cst.ClassDef):
            self.scopes.pop()
        elif isinstance(node, cst.Lambda):
            self.scopes.pop()
        elif isinstance(node, (cst.ListComp, cst.SetComp, cst.DictComp, cst.GeneratorExp)):
            self.scopes.pop()
        elif isinstance(node, (cst.For, cst.While)):
            self.loops.pop()
        elif isinstance(node, cst.With):
            if with_disables_grad(node):
                self._no_grad_depth -= 1


def target_names(node: cst.BaseExpression) -> List[str]:
    """Names bound by an assignment/for target, including tuple unpacking."""
    names: List[str] = []
    if isinstance(node, cst.Name):
        names.append(node.value)
    elif isinstance(node, (cst.Tuple, cst.List)):
        for element in node.elements:
            names.extend(target_names(element.value))
    elif isinstance(node, cst.StarredElement):
        names.extend(target_names(node.value))
    elif isinstance(node, cst.Attribute):
        dotted = dotted_name(node)
        if dotted:
            names.append(dotted)
    # Subscript targets (``losses[i] = x``) mutate an existing container rather than
    # creating one, so they deliberately do not count as a binding here.
    return names


def _assignment_target_names(node: cst.CSTNode) -> List[str]:
    names: List[str] = []
    if isinstance(node, cst.Assign):
        for target in node.targets:
            names.extend(target_names(target.target))
    elif isinstance(node, (cst.AugAssign, cst.AnnAssign)):
        names.extend(target_names(node.target))
    elif isinstance(node, cst.NamedExpr):
        names.extend(target_names(node.target))
    return names
