"""Static provider: bundled snapshot plus the analytic parameter formula.

Zero dependencies. This is the provider that runs in CI on every PR.
"""

from __future__ import annotations

from typing import Optional

from .. import archdb
from ..costmodel import params_from_transformer_shape
from ..types import Confidence, ModelKind, ModelProfile, TransformerShape


def resolve(reference: str) -> Optional[ModelProfile]:
    """Resolve a model name against the bundled snapshot."""
    return archdb.resolve(reference)


def from_shape(shape: TransformerShape, name: str = "custom") -> ModelProfile:
    """Build a profile from explicit architecture dimensions.

    Confidence is MEDIUM rather than HIGH: the formula is accurate to about 1% on the
    models we validated it against, but it does not know about a specific model's extra
    heads, adapters or bias terms.
    """
    params = params_from_transformer_shape(shape)
    return ModelProfile(
        name=name,
        param_count=params,
        trainable_params=params,
        source="formula",
        confidence=Confidence.MEDIUM,
        kind=ModelKind.TRANSFORMER,
        shape=shape,
    )


def from_param_count(params: int, name: str = "custom") -> ModelProfile:
    """User told us the parameter count directly (``--params 7B``)."""
    return ModelProfile(
        name=name,
        param_count=params,
        trainable_params=params,
        source="user-supplied",
        confidence=Confidence.MEDIUM,
        kind=ModelKind.UNKNOWN,
    )
