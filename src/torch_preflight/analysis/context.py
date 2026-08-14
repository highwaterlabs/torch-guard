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
    #: Names assigned the result of a sigmoid call, e.g. ``probs``.
    sigmoid_vars: Dict[str, str] = field(default_factory=dict)
    #: Any sigmoid at all in the file, call or layer. Its *absence* is the evidence that
    #: a value reaching ``BCELoss`` is raw logits rather than probabilities.
    has_sigmoid: bool = False
    uses_bce_with_logits: bool = False
    #: DDP or an explicit process group: the run has more than one rank.
    uses_distributed: bool = False
    #: Names bound to a ``DistributedSampler(...)``, so a sampler passed by variable counts.
    distributed_sampler_vars: Set[str] = field(default_factory=set)
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
        self.sigmoid_vars: Dict[str, str] = {}
        self.has_sigmoid = False
        self.uses_bce_with_logits = False
        self.uses_distributed = False
        self.distributed_sampler_vars: Set[str] = set()
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
            elif leaf in ("sigmoid", "Sigmoid"):
                for target in node.targets:
                    name = dotted_name(target.target)
                    if name:
                        self.sigmoid_vars[name] = "sigmoid"
            elif leaf == "DistributedSampler":
                for target in node.targets:
                    name = dotted_name(target.target)
                    if name:
                        self.distributed_sampler_vars.add(name)
        return True

    def visit_Call(self, node: cst.Call) -> bool:
        leaf = final_attr(node.func)
        if leaf in ("CrossEntropyLoss", "cross_entropy"):
            self.uses_cross_entropy = True
        elif leaf in ("NLLLoss", "nll_loss"):
            self.uses_nll_loss = True
        elif leaf in ("BCEWithLogitsLoss", "binary_cross_entropy_with_logits"):
            self.uses_bce_with_logits = True
        if leaf in ("sigmoid", "Sigmoid"):
            self.has_sigmoid = True
        if leaf in ("DistributedDataParallel", "DDP", "init_process_group"):
            self.uses_distributed = True
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
        sigmoid_vars=facts.sigmoid_vars,
        has_sigmoid=facts.has_sigmoid or ".sigmoid()" in source,
        uses_bce_with_logits=facts.uses_bce_with_logits,
        uses_distributed=facts.uses_distributed,
        distributed_sampler_vars=facts.distributed_sampler_vars,
        is_lightning=facts.is_lightning,
        uses_cuda=any(marker in source for marker in _CUDA_MARKERS),
    )


__all__ = ["FileContext", "build_context", "PositionProvider", "SOFTMAX_NAMES"]
