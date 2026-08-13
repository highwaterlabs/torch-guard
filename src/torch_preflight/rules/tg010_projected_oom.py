"""TG010 - the configured training run will not fit the target GPU.

This is the rule that turns pre-flight estimation into a CI gate. It stays completely
silent unless the project declares a target:

    [tool.torch-preflight]
    target_gpu = "rtx4090"

and it never fires on a low-confidence estimate — a build failed by a guessed parameter
count would be far worse than no check at all (RFC 0001 §8).
"""

from __future__ import annotations

import libcst as cst

from ..diagnostics import Category, Severity
from ..vram import hardware
from ..vram.costmodel import estimate
from ..vram.extract import extract
from ..vram.providers import resolve_profile
from ..vram.solver import solve
from ..vram.types import Confidence, format_bytes
from .base import Rule, register


@register
class ProjectedOom(Rule):
    code = "TG010"
    name = "projected-oom"
    summary = "Projected peak VRAM exceeds the configured target GPU"
    severity = Severity.ERROR
    category = Category.CRITICAL_OOM
    explanation = """
torch-preflight reads the training configuration out of the script — model, batch size,
sequence length, precision, optimizer, sharding — and projects peak VRAM against the GPU
declared in ``[tool.torch-preflight] target_gpu``. When the projection exceeds that card, the
run is going to fail, and it is far cheaper to learn that in CI than forty minutes into a
rented A100.

The rule is deliberately conservative:

* it does nothing unless ``target_gpu`` is set;
* it needs a model it can identify — a ``from_pretrained("...")`` literal, or an entry in
  the bundled architecture snapshot;
* it never fires on a LOW or UNKNOWN confidence estimate;
* it never reaches the network.

Run ``torch-preflight estimate <script> --gpu <target>`` for the full breakdown and the list
of changes that would make it fit.
""".strip()

    @classmethod
    def should_run(cls, ctx, cfg) -> bool:
        """Without a resolvable target there is nothing to compare against.

        Worth a real pre-check: this rule runs a second full extraction pass over the
        file, so skipping it when unconfigured is the difference between one traversal
        and two on every file in the repo.
        """
        target = getattr(cfg, "target_gpu", None) if cfg is not None else None
        return bool(target) and hardware.resolve(target)[0] is not None

    def leave_Module(self, original_node: cst.Module) -> None:
        cfg = self.cfg
        gpu, count = hardware.resolve(cfg.target_gpu)

        extracted = extract(self.ctx)
        if extracted.model_ref is None:
            # Nothing identifiable to estimate from. Staying silent beats guessing.
            return

        # Bundled snapshot only: `check` must never hit the network (RFC 0001 §4).
        profile = resolve_profile(extracted.model_ref, allow_network=False)
        if profile.confidence in (Confidence.UNKNOWN, Confidence.LOW):
            return

        report = estimate(profile, extracted.config, gpu, count)
        if not report.band.is_failure:
            return

        report.remediations = solve(report, limit=3)
        line = extracted.model_ref_line or 1
        utilization = (report.utilization or 0) * 100

        hint = f"Run `torch-preflight estimate {self.ctx.path} --gpu {cfg.target_gpu}` for the "
        fitting = next((r for r in report.remediations if r.fits), None)
        if fitting is not None:
            hint += f"full breakdown. Smallest change that fits: {fitting.label}."
        else:
            hint += "full breakdown. Nothing short of a bigger GPU fits this configuration."

        self.report_at_line(
            line,
            f"Projected peak VRAM is {format_bytes(report.total)} for {profile.name} at "
            f"batch {extracted.config.batch_size}"
            + (f" x seq {extracted.config.seq_len}" if extracted.config.seq_len else "")
            + f", which is {utilization:.0f}% of a {gpu.name} "
            f"({gpu.usable_gib:.1f} GiB usable). This run will OOM.",
            hint=hint,
        )
