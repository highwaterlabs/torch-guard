"""Per-file facts every rule can consult without re-walking the tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from .helpers import dotted_name, final_attr
from .provenance import Provenance, analyze

SOFTMAX_NAMES = frozenset({"softmax", "log_softmax", "Softmax", "LogSoftmax"})

_CUDA_MARKERS = (
    ".cuda(",
    '"cuda"',
    "'cuda'",
    "torch.device",
    ".to(device",
    "device=device",
    "torch.cuda",
    "DistributedDataParallel",
    "accelerator",
    "local_rank",
)


@dataclass
class FileContext:
    """Everything a rule needs to know about the file it is looking at."""

    path: str
    source: str
    module: cst.Module
    wrapper: MetadataWrapper
    provenance: Provenance

    imports: Set[str] = field(default_factory=set)
    #: Names assigned the result of a softmax/log_softmax call, e.g. ``probs``.
    softmax_vars: Dict[str, str] = field(default_factory=dict)
    uses_cross_entropy: bool = False
    uses_nll_loss: bool = False
    is_lightning: bool = False
    uses_cuda: bool = False

    @property
    def has_torch_import(self) -> bool:
        return any(i == "torch" or i.startswith("torch.") for i in self.imports)


class _FactCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.imports: Set[str] = set()
        self.softmax_vars: Dict[str, str] = {}
        self.uses_cross_entropy = False
        self.uses_nll_loss = False
        self.is_lightning = False

    def visit_Import(self, node: cst.Import) -> bool:
        for alias in node.names:
            name = dotted_name(alias.name)
            if name:
                self.imports.add(name)
        return False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        module = dotted_name(node.module) if node.module else ""
        if module:
            self.imports.add(module)
            if "lightning" in module:
                self.is_lightning = True
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        for base in node.bases:
            if "LightningModule" in (dotted_name(base.value) or ""):
                self.is_lightning = True
        return True

    def visit_Assign(self, node: cst.Assign) -> bool:
        value = node.value
        if isinstance(value, cst.Call):
            leaf = final_attr(value.func)
            if leaf in ("softmax", "log_softmax"):
                for target in node.targets:
                    name = dotted_name(target.target)
                    if name:
                        self.softmax_vars[name] = leaf
        return True

    def visit_Call(self, node: cst.Call) -> bool:
        leaf = final_attr(node.func)
        if leaf in ("CrossEntropyLoss", "cross_entropy"):
            self.uses_cross_entropy = True
        elif leaf in ("NLLLoss", "nll_loss"):
            self.uses_nll_loss = True
        return True


def build_context(path: str, source: str, module: Optional[cst.Module] = None) -> FileContext:
    """Parse ``source`` and collect all per-file facts.

    ``unsafe_skip_copy`` keeps the wrapper pointing at the *same* module object we hand
    to the fixer, so node identity survives between analysis and rewriting.
    """
    module = module if module is not None else cst.parse_module(source)
    wrapper = MetadataWrapper(module, unsafe_skip_copy=True)

    facts = _FactCollector()
    module.visit(facts)

    return FileContext(
        path=path,
        source=source,
        module=module,
        wrapper=wrapper,
        provenance=analyze(module),
        imports=facts.imports,
        softmax_vars=facts.softmax_vars,
        uses_cross_entropy=facts.uses_cross_entropy,
        uses_nll_loss=facts.uses_nll_loss,
        is_lightning=facts.is_lightning,
        uses_cuda=any(marker in source for marker in _CUDA_MARKERS),
    )


__all__ = ["FileContext", "build_context", "PositionProvider", "SOFTMAX_NAMES"]
