"""Model autodetection: resolve a locally defined model without being told where it is.

This is RFC 0001 §5 layer 2. Layer 1 (a ``from_pretrained("...")`` string literal) needs
no execution at all and lives in :mod:`extract`. Layer 3 is the honest failure message.

The job here is to turn

    model = Classifier(num_classes=10, width=2)

into an entry point the meta provider can instantiate, by:

1. locating the construction site in the CST;
2. constant-folding its arguments (literals and module-level constants);
3. working out which module defines the class, from the file's own imports.

Safety
------
Resolving a class means **importing the module that defines it**, which executes that
module's top level. Two rules keep that from being reckless:

* We never import the training script to *find* the model — we read it statically.
* If the class turns out to be defined in the script itself, the module is imported only
  when its top level is side-effect free: imports, defs, literal constants and an
  ``if __name__ == "__main__"`` guard. A script that builds a dataset or a model at module
  level is refused, because importing it would do that work for real.

Anything unresolvable is reported with the exact argument that blocked it, never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import libcst as cst
from libcst.metadata import PositionProvider

from ..analysis.context import FileContext
from ..analysis.helpers import dotted_name, final_attr
from ..analysis.provenance import MODEL_NAME_HINTS
from ..analysis.scope import target_names


@dataclass
class ModelConstruction:
    """A ``model = SomeClass(...)`` site found in the source."""

    class_name: str
    line: int
    kwargs: Dict[str, Any] = field(default_factory=dict)
    #: Rendered source of arguments we could not fold, e.g. ``cfg.num_classes``.
    unresolved: List[str] = field(default_factory=list)
    #: Module the class was imported from; None means it is defined in this file.
    module: Optional[str] = None
    #: True when the class is a local ``nn.Module`` subclass rather than a name guess.
    from_local_class: bool = False

    @property
    def resolvable(self) -> bool:
        return not self.unresolved


def fold_literal(node: cst.BaseExpression, constants: Dict[str, Any]) -> Any:
    """Evaluate a constructor argument statically, or raise ValueError.

    Deliberately narrow: literals, module-level constants bound to literals, negation and
    literal collections. Anything that would need to run code is refused.
    """
    if isinstance(node, cst.Integer):
        return int(node.value)
    if isinstance(node, cst.Float):
        return float(node.value)
    if isinstance(node, cst.SimpleString):
        return node.raw_value if hasattr(node, "raw_value") else node.value.strip("\"'")
    if isinstance(node, cst.Name):
        if node.value == "True":
            return True
        if node.value == "False":
            return False
        if node.value == "None":
            return None
        if node.value in constants:
            return constants[node.value]
        raise ValueError(node.value)
    if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Minus):
        return -fold_literal(node.expression, constants)
    if isinstance(node, (cst.Tuple, cst.List)):
        values = [fold_literal(e.value, constants) for e in node.elements]
        return tuple(values) if isinstance(node, cst.Tuple) else values

    raise ValueError(_render(node))


def _render(node: cst.CSTNode) -> str:
    try:
        return cst.Module(body=[]).code_for_node(node).strip()
    except Exception:
        return type(node).__name__


class _Finder(cst.CSTVisitor):
    """Collects module-level constants, imports, and model construction sites."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, ctx: FileContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.constants: Dict[str, Any] = {}
        #: class name -> module it was imported from
        self.imported_from: Dict[str, str] = {}
        self.sites: List[ModelConstruction] = []
        self._depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._depth += 1
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self._depth -= 1

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        module = dotted_name(node.module) if node.module else None
        if module is None or isinstance(node.names, cst.ImportStar):
            return False
        for alias in node.names:
            name = alias.asname.name.value if alias.asname else alias.name.value
            if isinstance(name, str):
                self.imported_from[name] = module
        return False

    def visit_Assign(self, node: cst.Assign) -> bool:
        # Module-level literal constants are foldable into constructor arguments.
        if self._depth == 0:
            try:
                value = fold_literal(node.value, self.constants)
            except ValueError:
                value = _UNSET
            if value is not _UNSET:
                for target in node.targets:
                    for name in target_names(target.target):
                        self.constants[name] = value

        if not isinstance(node.value, cst.Call):
            return True

        class_name = final_attr(node.value.func)
        if not class_name:
            return True

        targets = [n for t in node.targets for n in target_names(t.target)]
        local_class = class_name in self.ctx.provenance.module_classes
        name_hint = any(t.rsplit(".", 1)[-1] in MODEL_NAME_HINTS for t in targets)
        if not (local_class or name_hint):
            return True

        site = ModelConstruction(
            class_name=class_name,
            line=self.get_metadata(PositionProvider, node).start.line,
            module=self.imported_from.get(class_name),
            from_local_class=local_class,
        )
        for arg in node.value.args:
            try:
                folded = fold_literal(arg.value, self.constants)
            except ValueError as exc:
                site.unresolved.append(
                    f"{arg.keyword.value}={exc}" if arg.keyword else str(exc)
                )
                continue
            if arg.keyword is not None:
                site.kwargs[arg.keyword.value] = folded
        self.sites.append(site)
        return True


_UNSET = object()


def find_model_construction(ctx: FileContext) -> Optional[ModelConstruction]:
    """Best candidate model construction site in a file, or None."""
    finder = _Finder(ctx)
    ctx.wrapper.visit(finder)
    if not finder.sites:
        return None

    # A class we can see subclassing nn.Module beats a name-based guess.
    finder.sites.sort(key=lambda s: (not s.from_local_class, bool(s.unresolved)))
    return finder.sites[0]


def module_is_import_safe(module: cst.Module) -> bool:
    """True when importing this module would not do real work.

    Allows imports, ``def``/``class``, docstrings, literal constants and an
    ``if __name__ == "__main__"`` guard. A module-level call — building a dataset, a model,
    or connecting to something — makes it unsafe, because importing runs it.
    """
    for statement in module.body:
        if isinstance(statement, (cst.FunctionDef, cst.ClassDef)):
            continue
        if isinstance(statement, cst.If):
            if _is_main_guard(statement):
                continue
            return False
        if not isinstance(statement, cst.SimpleStatementLine):
            return False
        for small in statement.body:
            if isinstance(small, (cst.Import, cst.ImportFrom, cst.Pass)):
                continue
            if isinstance(small, cst.Expr) and isinstance(small.value, cst.SimpleString):
                continue  # docstring
            if isinstance(small, (cst.Assign, cst.AnnAssign)):
                value = small.value
                if value is not None and _contains_call(value):
                    return False
                continue
            return False
    return True


def _is_main_guard(node: cst.If) -> bool:
    test = node.test
    if not isinstance(test, cst.Comparison):
        return False
    return dotted_name(test.left) == "__name__"


def _contains_call(node: cst.CSTNode) -> bool:
    finder = _CallFinder()
    node.visit(finder)
    return finder.found


class _CallFinder(cst.CSTVisitor):
    def __init__(self) -> None:
        self.found = False

    def visit_Call(self, node: cst.Call) -> bool:
        self.found = True
        return False


@dataclass
class Autodetected:
    reference: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None
    construction: Optional[ModelConstruction] = None

    @property
    def ok(self) -> bool:
        return self.reference is not None


def autodetect(ctx: FileContext) -> Autodetected:
    """Find and resolve the model defined or used by this file."""
    site = find_model_construction(ctx)
    if site is None:
        return Autodetected(reason=explain_failure(None, ctx.path))

    if site.unresolved:
        return Autodetected(reason=explain_failure(site, ctx.path), construction=site)

    if site.module:
        return Autodetected(
            reference=f"{site.module}:{site.class_name}",
            kwargs=site.kwargs,
            construction=site,
        )

    # Defined in the file under analysis. Importing it is only acceptable if its top
    # level does no real work.
    if module_is_import_safe(ctx.module):
        return Autodetected(
            reference=f"{ctx.path}:{site.class_name}",
            kwargs=site.kwargs,
            construction=site,
        )

    return Autodetected(reason=explain_failure(site, ctx.path), construction=site)


def explain_failure(construction: Optional[ModelConstruction], path: str) -> str:
    """The message shown when autodetection cannot produce an answer."""
    if construction is None:
        return (
            f"No model construction found in {path}. torch-guard reads architectures from "
            f'`from_pretrained("...")` literals and from local `nn.Module` subclasses; '
            f"anything else needs --model or --params."
        )
    if construction.unresolved:
        blocked = ", ".join(construction.unresolved)
        return (
            f"Could not resolve the model automatically.\n"
            f"  {path}:{construction.line}  {construction.class_name}({blocked})\n"
            f"  Those arguments are computed at runtime, so they cannot be read statically.\n"
            f"  Pass them explicitly:  --model <module>:{construction.class_name} "
            f"--model-args key=value"
        )
    return (
        f"Found {construction.class_name} in {path}, but it is defined in a module whose "
        f"top level does work on import (it builds objects or calls functions), so "
        f"importing it to measure the model would run that work.\n"
        f"  Move the construction into a function, or pass "
        f"--model <module>:{construction.class_name} explicitly."
    )
