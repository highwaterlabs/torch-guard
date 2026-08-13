"""Terminal and JSON rendering for VRAM reports."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from rich.console import Console
from rich.text import Text

from ..vram.types import GIB, RiskBand, VramReport, format_bytes

BAND_STYLE = {
    RiskBand.FITS: "bold green",
    RiskBand.TIGHT: "bold yellow",
    RiskBand.LIKELY_OOM: "bold red",
    RiskBand.CERTAIN_OOM: "bold red",
    RiskBand.UNKNOWN: "bold dim",
}

BAND_LABEL = {
    RiskBand.FITS: "✓ FITS",
    RiskBand.TIGHT: "! TIGHT",
    RiskBand.LIKELY_OOM: "✗ LIKELY OOM",
    RiskBand.CERTAIN_OOM: "✗ OOM",
    RiskBand.UNKNOWN: "? UNKNOWN",
}


def _human_params(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1e9:.2f} B"
    if count >= 1_000_000:
        return f"{count / 1e6:.1f} M"
    return str(count)


def render_terminal(report: VramReport, console: Optional[Console] = None) -> None:
    console = console or Console()

    if not report.profile.resolved:
        console.print()
        console.print(Text("Could not estimate VRAM", style="bold yellow"))
        console.print()
        for line in (report.profile.reason or "").splitlines():
            console.print(Text("  " + line))
        console.print()
        return

    profile = report.profile
    config = report.config

    console.print()
    console.print(
        Text("Model      ", style="dim")
        + Text(profile.name, style="bold")
        + Text(f"  ({profile.source})", style="dim")
        + Text(f"   {_human_params(profile.param_count)} params")
    )
    console.print(Text("Config     ", style="dim") + Text(_config_line(config)))
    if config.sources:
        shown = ", ".join(f"{k}={v}" for k, v in sorted(config.sources.items()))
        console.print(Text("Read from  ", style="dim") + Text(shown, style="dim"))
    console.print()

    for label, value in report.breakdown.items():
        if value <= 0:
            continue
        bar = _bar(value, report.total)
        console.print(
            Text(f"  {label:<18}", style="dim")
            + Text(f"{format_bytes(value):>10}  ")
            + Text(bar, style="cyan")
        )

    console.print(Text("  " + "─" * 46, style="dim"))
    low, high = report.interval
    console.print(
        Text(f"  {'projected peak':<18}", style="bold")
        + Text(f"{format_bytes(report.total):>10}", style="bold")
        + Text(f"   ({format_bytes(low)} – {format_bytes(high)})", style="dim")
    )

    if report.gpu is not None:
        console.print()
        utilization = report.utilization or 0.0
        console.print(
            Text("Target     ", style="dim")
            + Text(f"{report.gpu.name} ({report.gpu.usable_gib:.1f} GiB usable)")
            + Text(f"   →   {utilization * 100:.0f}% of capacity   ")
            + Text(BAND_LABEL[report.band], style=BAND_STYLE[report.band])
        )

    if report.remediations:
        console.print()
        console.print(Text("What would make it fit:", style="bold"))
        # Numbers first in fixed columns, label last, so an arbitrarily long combined
        # label can never break the alignment.
        usable = report.gpu.usable_bytes if report.gpu is not None else None
        for item in report.remediations:
            # Three states, because "the point estimate fits but the error margin does
            # not" is genuinely different from "this does not fit at all".
            if item.fits:
                mark, style = "✓", "green"
            elif usable is not None and item.new_total < usable:
                mark, style = "~", "yellow"
            else:
                mark, style = "✗", "dim"
            console.print(
                Text(f"  {mark}  ", style=style)
                + Text(f"−{format_bytes(item.saved_bytes):>10}")
                + Text(f"  →  {format_bytes(item.new_total):>10}   ")
                + Text(item.label, style=style)
            )
            if item.note:
                console.print(Text(f"       {item.note}", style="dim"))

    if report.notes:
        console.print()
        for note in report.notes:
            console.print(Text("  note: ", style="yellow") + Text(note, style="dim"))

    console.print()


def _config_line(config) -> str:
    parts = [
        config.precision.value,
        config.optimizer.label,
        f"batch {config.batch_size}",
    ]
    if config.seq_len:
        parts.append(f"seq {config.seq_len}")
    if config.image_size:
        parts.append(f"image {config.image_size}px")
    if config.accumulation_steps > 1:
        parts.append(f"{config.accumulation_steps}x accumulation")
    if config.gradient_checkpointing:
        parts.append("checkpointing")
    if config.flash_attention:
        parts.append("flash attention")
    if config.sharding.value != "none":
        parts.append(f"{config.sharding.label} x{config.world_size}")
    if config.inference_only:
        parts.append("inference only (no backward found)")
    return " · ".join(parts)


def _bar(value: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return ""
    filled = int(round(width * value / total))
    return "█" * max(filled, 1 if value else 0)


def render_json(report: VramReport) -> str:
    payload: Dict[str, Any] = {
        "version": 1,
        "resolved": report.profile.resolved,
        "model": {
            "name": report.profile.name,
            "params": report.profile.param_count,
            "source": report.profile.source,
            "confidence": report.profile.confidence.value,
            "reason": report.profile.reason,
        },
        "config": {
            "batch_size": report.config.batch_size,
            "seq_len": report.config.seq_len,
            "image_size": report.config.image_size,
            "precision": report.config.precision.value,
            "optimizer": report.config.optimizer.value,
            "gradient_checkpointing": report.config.gradient_checkpointing,
            "flash_attention": report.config.flash_attention,
            "accumulation_steps": report.config.accumulation_steps,
            "world_size": report.config.world_size,
            "sharding": report.config.sharding.value,
            "global_batch": report.config.global_batch,
            "inference_only": report.config.inference_only,
            "sources": report.config.sources,
        },
        "breakdown": {
            name.replace(" ", "_"): value for name, value in report.breakdown.items()
        },
        "total_bytes": report.total,
        "interval_bytes": list(report.interval),
        "band": report.band.value,
        "notes": report.notes,
    }

    if report.gpu is not None:
        payload["gpu"] = {
            "name": report.gpu.name,
            "key": report.gpu.key,
            "usable_bytes": report.gpu.usable_bytes,
            "count": report.gpu_count,
            "utilization": report.utilization,
        }

    payload["remediations"] = [
        {
            "label": item.label,
            "saved_bytes": item.saved_bytes,
            "new_total_bytes": item.new_total,
            "fits": item.fits,
            "note": item.note,
        }
        for item in report.remediations
    ]
    return json.dumps(payload, indent=2)
