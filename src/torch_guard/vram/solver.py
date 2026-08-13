"""Remediation search: what change would make this fit?

Everyone else stops at the number. This is the part an engineer actually wants at 2am, and
because it is pure arithmetic on the cost model it ships in the free tier.

Candidates are ranked by whether they fit first, then by how much they disturb the training
recipe — a change that is mathematically equivalent (flash attention) beats one that alters
numerics (bf16), which beats one that changes the method entirely (LoRA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from .costmodel import estimate
from .types import (
    ModelProfile,
    OptimizerKind,
    PrecisionMode,
    Remediation,
    RunConfig,
    Sharding,
    VramReport,
)


@dataclass
class _Candidate:
    label: str
    apply: Callable[[RunConfig], Optional[RunConfig]]
    disruption: int
    note: str = ""
    #: Candidates sharing a group are mutually exclusive — you cannot switch the
    #: optimizer to both 8-bit AdamW and Adafactor. The greedy stack keeps the first
    #: (least disruptive) one it reaches and skips the rest of the group.
    group: str = ""


def _checkpointing(config: RunConfig) -> Optional[RunConfig]:
    if config.gradient_checkpointing:
        return None
    return config.replace(gradient_checkpointing=True)


def _flash(config: RunConfig) -> Optional[RunConfig]:
    if config.flash_attention:
        return None
    return config.replace(flash_attention=True)


def _eight_bit(config: RunConfig) -> Optional[RunConfig]:
    if config.optimizer not in (OptimizerKind.ADAM, OptimizerKind.ADAMW):
        return None
    return config.replace(optimizer=OptimizerKind.ADAM_8BIT)


def _adafactor(config: RunConfig) -> Optional[RunConfig]:
    if config.optimizer is OptimizerKind.ADAFACTOR:
        return None
    return config.replace(optimizer=OptimizerKind.ADAFACTOR)


def _amp(config: RunConfig) -> Optional[RunConfig]:
    if config.precision is not PrecisionMode.FP32:
        return None
    return config.replace(precision=PrecisionMode.AMP)


def _pure_bf16(config: RunConfig) -> Optional[RunConfig]:
    if config.precision in (PrecisionMode.PURE_BF16, PrecisionMode.PURE_FP16):
        return None
    return config.replace(precision=PrecisionMode.PURE_BF16)


def _halve_batch(factor: int) -> Callable[[RunConfig], Optional[RunConfig]]:
    def apply(config: RunConfig) -> Optional[RunConfig]:
        if config.batch_size < factor * 2:
            return None
        return config.replace(
            batch_size=config.batch_size // factor,
            accumulation_steps=config.accumulation_steps * factor,
        )

    return apply


def _shard(world: int) -> Callable[[RunConfig], Optional[RunConfig]]:
    def apply(config: RunConfig) -> Optional[RunConfig]:
        if config.sharding is Sharding.ZERO3 and config.world_size >= world:
            return None
        return config.replace(sharding=Sharding.ZERO3, world_size=world)

    return apply


def _lora(config: RunConfig) -> Optional[RunConfig]:
    if config.frozen_fraction > 0.5:
        return None
    return config.replace(frozen_fraction=0.99)


CANDIDATES: List[_Candidate] = [
    _Candidate("flash attention / SDPA", _flash, 0,
               "mathematically equivalent, removes the O(seq²) attention term",
               group="attention"),
    _Candidate("gradient checkpointing", _checkpointing, 1,
               "same result, roughly 30% slower", group="checkpointing"),
    _Candidate("8-bit AdamW (bitsandbytes)", _eight_bit, 1,
               "quantised optimizer state, minimal quality impact", group="optimizer"),
    _Candidate("Adafactor optimizer", _adafactor, 2,
               "factored second moments; changes the update rule", group="optimizer"),
    _Candidate("mixed precision (autocast)", _amp, 2,
               "shrinks activations only; weights stay fp32", group="precision"),
    _Candidate("pure bf16 weights", _pure_bf16, 3,
               "halves weights and gradients; watch for numerical drift", group="precision"),
    _Candidate("halve micro-batch (2x accumulation)", _halve_batch(2), 2,
               "same global batch, one extra forward per step", group="batch"),
    _Candidate("quarter micro-batch (4x accumulation)", _halve_batch(4), 3,
               "same global batch, more steps per update", group="batch"),
    _Candidate("FSDP / ZeRO-3 across 2 GPUs", _shard(2), 3, "requires 2 devices", group="sharding"),
    _Candidate("FSDP / ZeRO-3 across 4 GPUs", _shard(4), 4, "requires 4 devices", group="sharding"),
    _Candidate("FSDP / ZeRO-3 across 8 GPUs", _shard(8), 4, "requires 8 devices", group="sharding"),
    _Candidate("LoRA / freeze the backbone", _lora, 5,
               "no gradients or optimizer state for frozen weights", group="freeze"),
]

_BY_LABEL = {c.label: c for c in CANDIDATES}


def _evaluate(
    profile: ModelProfile, config: RunConfig, gpu, baseline_total: int
) -> Optional[Remediation]:
    report = estimate(profile, config, gpu)
    usable = gpu.usable_bytes if gpu is not None else None
    return Remediation(
        label="",
        saved_bytes=baseline_total - report.total,
        new_total=report.total,
        fits=usable is not None and report.interval[1] < usable,
    )


def solve(report: VramReport, limit: int = 6) -> List[Remediation]:
    """Find configuration changes that reduce peak memory, best-first."""
    if not report.profile.resolved or report.gpu is None:
        return []

    profile = report.profile
    baseline = report.config
    baseline_total = report.total
    results: List[Remediation] = []

    for candidate in CANDIDATES:
        changed = candidate.apply(baseline)
        if changed is None:
            continue
        outcome = _evaluate(profile, changed, report.gpu, baseline_total)
        if outcome is None or outcome.saved_bytes <= 0:
            continue
        outcome.label = candidate.label
        outcome.disruption = candidate.disruption
        outcome.note = candidate.note
        results.append(outcome)

    # If no single change is enough, stack them cheapest-first until something fits.
    if not any(r.fits for r in results):
        stacked = _greedy_stack(profile, baseline, report.gpu, baseline_total)
        if stacked is not None:
            results.append(stacked)

    results.sort(key=lambda r: (not r.fits, r.disruption, -r.saved_bytes))
    return results[:limit]


#: Beyond this many stacked changes the advice stops being advice. If a configuration
#: needs more, the honest answer is that it does not fit this GPU.
MAX_STACK = 4


def _greedy_stack(
    profile: ModelProfile, baseline: RunConfig, gpu, baseline_total: int
) -> Optional[Remediation]:
    """Apply changes in increasing order of disruption until the config fits.

    This is what makes the advice actionable when a single knob is not enough: it finds
    the least-disruptive *combination* rather than leaving the user to work it out.
    Capped at :data:`MAX_STACK`: "change these seven things" is not a recommendation.
    """
    config = baseline
    applied: List[str] = []
    used_groups: set = set()
    disruption = 0

    for candidate in sorted(CANDIDATES, key=lambda c: c.disruption):
        if candidate.group and candidate.group in used_groups:
            continue
        changed = candidate.apply(config)
        if changed is None:
            continue

        trial = estimate(profile, changed, gpu)
        current = estimate(profile, config, gpu)
        if trial.total >= current.total:
            continue  # this knob does not help on top of what we already applied

        config = changed
        applied.append(candidate.label)
        if candidate.group:
            used_groups.add(candidate.group)
        disruption = max(disruption, candidate.disruption)

        if len(applied) >= MAX_STACK and trial.interval[1] >= gpu.usable_bytes:
            break

        if trial.interval[1] < gpu.usable_bytes:
            return Remediation(
                label=" + ".join(applied),
                saved_bytes=baseline_total - trial.total,
                new_total=trial.total,
                fits=True,
                disruption=disruption + 1,
                note="smallest combination that fits",
            )

    if not applied:
        return None

    final = estimate(profile, config, gpu)
    if final.total < gpu.usable_bytes:
        note = (
            "lands inside the card but not inside the error margin — it may well run, "
            "but there is no headroom for a longer batch or a fragmented allocator"
        )
    else:
        note = (
            "even stacked together these do not fit; this needs a larger GPU, more "
            "devices, or a parameter-efficient method such as LoRA"
        )
    return Remediation(
        label=" + ".join(applied),
        saved_bytes=baseline_total - final.total,
        new_total=final.total,
        fits=False,
        disruption=disruption + 1,
        note=note,
    )
