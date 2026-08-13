"""Pre-flight VRAM estimation (RFC 0001).

Public entry point:

    from torch_guard.vram import estimate_script
    report = estimate_script("train.py", gpu="a100-80gb")

Nothing in this package imports torch or huggingface_hub at module level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..analysis.context import build_context
from . import archdb, hardware
from . import autodetect
from .costmodel import estimate
from .extract import ExtractedConfig, extract, extract_from_source
from .providers import resolve_profile
from .solver import solve
from .types import (
    Confidence,
    MemoryBreakdown,
    ModelProfile,
    OptimizerKind,
    PrecisionMode,
    Remediation,
    RiskBand,
    RunConfig,
    Sharding,
    TransformerShape,
    VramReport,
    format_bytes,
)


def _apply_overrides(config: RunConfig, overrides: Dict[str, Any]) -> RunConfig:
    changes = {key: value for key, value in overrides.items() if value is not None}
    for key in changes:
        config.sources[key] = "command line"
    return config.replace(**changes) if changes else config


def _unresolved_reason(extracted: ExtractedConfig, path: str) -> str:
    """Only reached when a ``from_pretrained`` call had a non-literal argument."""
    expression, line = extracted.unresolved_models[0]
    return (
        f"Could not resolve the model automatically.\n"
        f"  {path}:{line}  from_pretrained({expression})\n"
        f"  The argument is computed at runtime, so it cannot be read statically.\n"
        f"  Pass it explicitly:  --model <architecture-name>"
    )


def _resolve_target(gpu: Optional[str], gpu_memory: Optional[int]):
    """Resolve the target device, refusing to silently ignore an unknown name.

    Quietly dropping an unrecognised --gpu would hand back a report that looks checked
    but was never compared against anything.
    """
    device, count = (None, 1)
    if gpu:
        device, count = hardware.resolve(gpu)
        if device is None and gpu_memory is None:
            raise ValueError(
                f"unknown GPU {gpu!r} — run `torch-guard gpus` for the list, or pass "
                f"--gpu-memory 48GiB"
            )
    if gpu_memory is not None:
        device = hardware.custom_gpu(gpu_memory, name=gpu or "custom")
    return device, count


def estimate_script(
    path: str,
    source: Optional[str] = None,
    *,
    gpu: Optional[str] = None,
    gpu_memory: Optional[int] = None,
    model: Optional[str] = None,
    online: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
    model_args: Optional[Dict[str, Any]] = None,
    with_remediations: bool = True,
) -> VramReport:
    """Estimate peak VRAM for a training script."""
    if source is None:
        source = Path(path).read_text(encoding="utf-8")

    ctx = build_context(path, source)
    extracted = extract(ctx)
    config = _apply_overrides(extracted.config, overrides or {})

    reference = model or extracted.model_ref
    if reference:
        # Layer 1: an explicit name, or a from_pretrained("...") literal. No execution.
        profile = resolve_profile(
            reference, allow_network=online, config=config, model_args=model_args
        )
    elif extracted.unresolved_models:
        profile = ModelProfile.unknown("<unknown>", _unresolved_reason(extracted, path))
    else:
        # Layer 2: a locally defined model. Resolves the class from the file's imports
        # and instantiates it on the meta device, refusing to import anything whose top
        # level would do real work.
        detected = autodetect.autodetect(ctx)
        if detected.ok:
            profile = resolve_profile(
                detected.reference,
                allow_network=online,
                config=config,
                model_args=model_args or detected.kwargs,
            )
        else:
            # Layer 3: say exactly what blocked resolution.
            profile = ModelProfile.unknown("<unknown>", detected.reason)

    device, count = _resolve_target(gpu, gpu_memory)

    report = estimate(profile, config, device, count)
    if with_remediations and report.band.is_failure:
        report.remediations = solve(report)
    return report


def estimate_config(
    profile: ModelProfile,
    config: RunConfig,
    *,
    gpu: Optional[str] = None,
    gpu_memory: Optional[int] = None,
    with_remediations: bool = True,
) -> VramReport:
    """Estimate from an already-built profile and config (no script involved)."""
    device, count = _resolve_target(gpu, gpu_memory)

    report = estimate(profile, config, device, count)
    if with_remediations and report.band.is_failure:
        report.remediations = solve(report)
    return report


__all__ = [
    "Confidence",
    "MemoryBreakdown",
    "ModelProfile",
    "OptimizerKind",
    "PrecisionMode",
    "Remediation",
    "RiskBand",
    "RunConfig",
    "Sharding",
    "TransformerShape",
    "VramReport",
    "archdb",
    "autodetect",
    "estimate",
    "estimate_config",
    "estimate_script",
    "format_bytes",
    "hardware",
    "resolve_profile",
    "solve",
]
