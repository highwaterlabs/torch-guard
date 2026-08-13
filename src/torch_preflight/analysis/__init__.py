"""Static analysis layer: scope tracking, grad provenance and per-file facts."""

from .context import FileContext, build_context
from .provenance import Provenance, analyze
from .scope import ScopeTrackingVisitor

__all__ = [
    "FileContext",
    "Provenance",
    "ScopeTrackingVisitor",
    "analyze",
    "build_context",
]
