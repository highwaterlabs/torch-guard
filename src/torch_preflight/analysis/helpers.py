"""Small, dependency-free helpers for reading libcst trees.

Rules are written against these so they never hand-roll attribute chain walking.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import libcst as cst

# Methods that sever the autograd graph. Storing the result of any of these is safe.
# ``numpy()`` is included because PyTorch refuses to call it on an attached tensor,
# so code that reaches it at runtime was already detached.
DETACHING_METHODS = frozenset(
    {"item", "detach", "tolist", "numpy", "clone_detached", "detach_"}
)

# Builtins that collapse a tensor to a Python scalar/structure.
DETACHING_BUILTINS = frozenset({"float", "int", "bool", "str", "len", "round"})

# Operations whose output carries no autograd graph at all: integer indices, boolean
# masks, shape queries and casts to integral dtypes. Storing these is always safe.
# Float-preserving casts (``.float()``, ``.half()``) are deliberately absent — they do
# keep the graph. So do ``.sum()``, ``.mean()``, ``.max()`` and friends.
NON_DIFFERENTIABLE_METHODS = frozenset(
    {
        "argmax", "argmin", "argsort", "argwhere", "nonzero", "bincount", "unique",
        "count_nonzero", "size", "numel", "dim", "ndimension", "element_size",
        "long", "int", "bool", "byte", "short", "char",
        "eq", "ne", "gt", "lt", "ge", "le", "isnan", "isinf", "isfinite",
    }
)

# Context managers / decorators that disable autograd.
NO_GRAD_NAMES = frozenset({"no_grad", "inference_mode"})


def dotted_name(node: cst.BaseExpression) -> Optional[str]:
    """Render an attribute chain as a dotted string, or None if it is not one.

    ``torch.nn.functional.relu`` -> ``"torch.nn.functional.relu"``
    ``foo().bar`` -> ``None`` (contains a call, not a pure name path)
    """
    parts: List[str] = []
    current: cst.BaseExpression = node
    while True:
        if isinstance(current, cst.Name):
            parts.append(current.value)
            return ".".join(reversed(parts))
        if isinstance(current, cst.Attribute):
            parts.append(current.attr.value)
            current = current.value
            continue
        return None


def call_func_name(node: cst.Call) -> Optional[str]:
    """Dotted name of a call's callee: ``F.cross_entropy(..)`` -> ``F.cross_entropy``."""
    return dotted_name(node.func)


def final_attr(node: cst.BaseExpression) -> Optional[str]:
    """Last segment of an attribute chain: ``a.b.c`` -> ``"c"``; ``a`` -> ``"a"``."""
    if isinstance(node, cst.Attribute):
        return node.attr.value
    if isinstance(node, cst.Name):
        return node.value
    return None


def base_name(node: cst.BaseExpression) -> Optional[str]:
    """Leftmost name of an expression: ``self.losses[0].x`` -> ``"self"``."""
    current: cst.BaseExpression = node
    while True:
        if isinstance(current, cst.Name):
            return current.value
        if isinstance(current, cst.Attribute):
            current = current.value
        elif isinstance(current, cst.Subscript):
            current = current.value
        elif isinstance(current, cst.Call):
            current = current.func
        else:
            return None


def method_call_receiver(node: cst.Call, method: str) -> Optional[cst.BaseExpression]:
    """If ``node`` is ``<expr>.<method>(...)``, return ``<expr>``."""
    func = node.func
    if isinstance(func, cst.Attribute) and func.attr.value == method:
        return func.value
    return None


def keyword_arg(node: cst.Call, name: str) -> Optional[cst.Arg]:
    for arg in node.args:
        if arg.keyword is not None and arg.keyword.value == name:
            return arg
    return None


def positional_args(node: cst.Call) -> List[cst.Arg]:
    return [a for a in node.args if a.keyword is None and a.star == ""]


def is_literal_zero(node: cst.BaseExpression) -> bool:
    return isinstance(node, cst.Integer) and node.value.strip() == "0"


def is_literal_true(node: cst.BaseExpression) -> bool:
    return isinstance(node, cst.Name) and node.value == "True"


def is_literal_false(node: cst.BaseExpression) -> bool:
    return isinstance(node, cst.Name) and node.value == "False"


def atomize(node: cst.BaseExpression) -> cst.BaseExpression:
    """Parenthesize ``node`` if appending ``.method()`` to it would rebind wrongly.

    ``a + b`` -> ``(a + b)`` so that ``.detach()`` applies to the sum, not to ``b``.
    """
    safe = (
        cst.Name,
        cst.Attribute,
        cst.Call,
        cst.Subscript,
        cst.Integer,
        cst.Float,
        cst.SimpleString,
        cst.Tuple,
        cst.List,
        cst.Dict,
    )
    if isinstance(node, safe):
        return node
    if node.lpar and node.rpar:
        return node
    return node.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])


def attach_method(node: cst.BaseExpression, method: str) -> cst.Call:
    """Build ``<node>.<method>()``, parenthesizing ``node`` when required."""
    receiver = atomize(node)
    # Strip trailing whitespace/comments that would land inside the new call.
    receiver = receiver.with_changes(
        lpar=list(receiver.lpar) if receiver.lpar else [],
        rpar=list(receiver.rpar) if receiver.rpar else [],
    )
    return cst.Call(
        func=cst.Attribute(value=receiver, attr=cst.Name(method)),
        args=[],
    )


def contains_call_to(
    tree: cst.CSTNode, method_names: Sequence[str], *, stop_at_nested_func: bool = False
) -> bool:
    """True if ``tree`` contains a ``<expr>.<name>(...)`` call for any given name."""
    finder = _MethodCallFinder(frozenset(method_names), stop_at_nested_func)
    tree.visit(finder)
    return finder.found


class _MethodCallFinder(cst.CSTVisitor):
    def __init__(self, names: frozenset, stop_at_nested_func: bool) -> None:
        self.names = names
        self.stop_at_nested_func = stop_at_nested_func
        self.found = False
        self._depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        if self.stop_at_nested_func:
            self._depth += 1
            if self._depth > 1:
                return False
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        if self.stop_at_nested_func:
            self._depth -= 1

    def visit_Call(self, node: cst.Call) -> Optional[bool]:
        if self.found:
            return False
        func = node.func
        if isinstance(func, cst.Attribute) and func.attr.value in self.names:
            self.found = True
        elif isinstance(func, cst.Name) and func.value in self.names:
            self.found = True
        return True
