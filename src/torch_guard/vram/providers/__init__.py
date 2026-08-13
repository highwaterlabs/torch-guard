"""Provider registry.

Providers are tried best-first and each one degrades to the next, so capability is
discovered at call time rather than declared at install time (RFC 0001 §4). Nothing here
imports an optional dependency at module level.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..types import ModelProfile, RunConfig
from . import static

#: Entry points need torch, which only the [vram] extra installs.
_NEEDS_VRAM_EXTRA = (
    "Profiling a Python entry point needs PyTorch: pip install 'torch-guard[vram]'. "
    "Alternatively pass a known architecture name (--model llama-2-7b) or --params."
)


def is_entry_point(reference: str) -> bool:
    """True for ``module.path:factory`` style references."""
    return ":" in reference and not reference.startswith(("http://", "https://"))


def resolve_profile(
    reference: str,
    *,
    allow_network: bool = False,
    config: Optional[RunConfig] = None,
    model_args: Optional[Dict[str, Any]] = None,
) -> ModelProfile:
    """Resolve a model reference to a profile, degrading honestly.

    Order: entry point via the meta device (exact, needs ``[vram]``) → bundled snapshot
    (free, offline) → hub (needs ``[hub]`` + network) → UNKNOWN.
    """
    if is_entry_point(reference):
        from . import meta  # local import keeps torch optional

        if not meta.available():
            return ModelProfile.unknown(reference, _NEEDS_VRAM_EXTRA)
        try:
            return meta.profile(reference, config, model_args)
        except meta.EntryPointError as exc:
            return ModelProfile.unknown(reference, str(exc))
        except Exception as exc:  # importing user code can fail in any way at all
            return ModelProfile.unknown(
                reference, f"{type(exc).__name__} while profiling {reference}: {exc}"
            )

    profile = static.resolve(reference)
    if profile is not None:
        return profile

    if allow_network:
        from . import hub  # local import keeps huggingface_hub optional

        if not hub.available():
            return ModelProfile.unknown(
                reference,
                f"{reference!r} is not in the bundled snapshot. Live lookup needs the hub "
                f"extra: pip install 'torch-guard[hub]'",
            )
        profile = hub.resolve(reference)
        if profile is not None:
            return profile
        return ModelProfile.unknown(
            reference,
            f"{reference!r} could not be resolved from the Hugging Face hub either "
            f"(not found, private, or no usable config.json).",
        )

    return ModelProfile.unknown(
        reference,
        f"{reference!r} is not in the bundled architecture snapshot. Retry with --online "
        f"to look it up on the Hugging Face hub, or pass --params explicitly.",
    )


__all__ = ["is_entry_point", "resolve_profile", "static"]
