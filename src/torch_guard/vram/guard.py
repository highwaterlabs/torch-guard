"""``VRAMGuard`` — fail at step 0 instead of OOM at step 400.

    from torch_guard import VRAMGuard

    with VRAMGuard(model, optimizer=optimizer, batch_size=32, seq_len=2048):
        train()

The static estimator answers "will this fit?" before the job is submitted. This answers it
*inside* the process, against the model that actually exists and the device actually
present — which catches the cases static analysis cannot see, like a model whose size
depends on a config file or a checkpoint that was loaded at runtime.

What it checks, and what it does not
------------------------------------
Parameters, gradients, optimizer state and the autocast weight cache are computed from the
live model and are **exact**. Activation memory needs a shape: pass ``seq_len`` or
``image_size`` and it is estimated, otherwise that term is reported as unknown and the
interval widens to say so.

Because of that, the guard **raises only when the run cannot fit even at the optimistic
end of the interval** (``CERTAIN_OOM``). Anything less certain is a warning. Aborting a
training job on a guess would be worse than the OOM it was trying to prevent; pass
``strict=True`` to raise on merely-likely failures too.

Requires the ``[vram]`` extra. ``torch`` is imported inside the methods.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

from . import hardware
from .costmodel import estimate
from .solver import solve
from .types import (
    Confidence,
    ModelKind,
    ModelProfile,
    OptimizerKind,
    PrecisionMode,
    RiskBand,
    RunConfig,
    VramReport,
    format_bytes,
)

#: Optimizer class name -> cost model kind.
_OPTIMIZER_CLASSES = {
    "sgd": OptimizerKind.SGD,
    "adam": OptimizerKind.ADAM,
    "adamw": OptimizerKind.ADAMW,
    "nadam": OptimizerKind.ADAMW,
    "radam": OptimizerKind.ADAMW,
    "adamw8bit": OptimizerKind.ADAM_8BIT,
    "adam8bit": OptimizerKind.ADAM_8BIT,
    "pagedadamw8bit": OptimizerKind.ADAM_8BIT,
    "adafactor": OptimizerKind.ADAFACTOR,
    "lion": OptimizerKind.LION,
    "lion8bit": OptimizerKind.LION_8BIT,
    "rmsprop": OptimizerKind.RMSPROP,
    "adagrad": OptimizerKind.ADAGRAD,
}


class VramRiskError(RuntimeError):
    """Raised when the configuration cannot fit the device."""

    def __init__(self, message: str, report: VramReport) -> None:
        super().__init__(message)
        self.report = report


def _infer_optimizer(optimizer) -> OptimizerKind:
    if optimizer is None:
        return OptimizerKind.ADAMW
    name = type(optimizer).__name__.lower().replace("_", "")
    kind = _OPTIMIZER_CLASSES.get(name, OptimizerKind.ADAMW)
    if kind is OptimizerKind.SGD:
        groups = getattr(optimizer, "param_groups", [])
        if any(group.get("momentum", 0) for group in groups):
            return OptimizerKind.SGD_MOMENTUM
    return kind


def _infer_precision(model) -> PrecisionMode:
    """Read the precision off the model's own parameters."""
    import torch

    for parameter in model.parameters():
        if parameter.dtype == torch.float16:
            return PrecisionMode.PURE_FP16
        if parameter.dtype == torch.bfloat16:
            return PrecisionMode.PURE_BF16
        break
    return PrecisionMode.FP32


def _current_device_gpu(model):
    """Describe the device the model lives on, using its real reported capacity."""
    import torch

    if not torch.cuda.is_available():
        return None

    index = 0
    for parameter in model.parameters():
        if parameter.is_cuda:
            index = parameter.device.index or 0
        break

    properties = torch.cuda.get_device_properties(index)
    gpu = hardware.custom_gpu(properties.total_memory, name=properties.name)
    # Prefer a measured context for a card we know about.
    known, _ = hardware.resolve(properties.name)
    if known is not None and known.context_mib:
        gpu = hardware.Gpu(
            key=known.key,
            name=properties.name,
            total_mib=properties.total_memory // (1024 ** 2),
            architecture=known.architecture,
            reserve_fraction=known.reserve_fraction,
            context_mib=known.context_mib,
        )
    return gpu


def profile_live_model(model, name: str = "<live model>") -> ModelProfile:
    """Exact parameter counts from an instantiated model.

    Shared parameters (tied embeddings) are counted once, matching how they occupy memory.
    """
    seen = set()
    total = trainable = 0
    for parameter in model.parameters():
        key = id(parameter)
        if key in seen:
            continue
        seen.add(key)
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()

    buffer_bytes = 0
    buffer_seen = set()
    for buffer in model.buffers():
        key = id(buffer)
        if key in buffer_seen:
            continue
        buffer_seen.add(key)
        buffer_bytes += buffer.numel() * buffer.element_size()

    return ModelProfile(
        name=name,
        param_count=total,
        trainable_params=trainable,
        source="live-model",
        confidence=Confidence.HIGH,
        kind=ModelKind.UNKNOWN,
        buffer_bytes=buffer_bytes,
    )


class VRAMGuard:
    """Context manager that refuses to start a run that cannot fit."""

    def __init__(
        self,
        model,
        *,
        optimizer=None,
        max_vram: Optional[str] = None,
        batch_size: int = 1,
        seq_len: Optional[int] = None,
        image_size: Optional[int] = None,
        precision: Optional[Any] = None,
        gradient_checkpointing: bool = False,
        flash_attention: bool = False,
        activation_bytes_per_sample: Optional[int] = None,
        strict: bool = False,
        verify: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.max_vram = max_vram
        self.strict = strict
        self.verify = verify

        self.config = RunConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            image_size=image_size,
            precision=PrecisionMode(precision) if precision else _infer_precision(model),
            optimizer=_infer_optimizer(optimizer),
            gradient_checkpointing=gradient_checkpointing,
            flash_attention=flash_attention,
        )
        self.profile = profile_live_model(model, type(model).__name__)
        if activation_bytes_per_sample is not None:
            self.profile.activation_bytes_per_sample = activation_bytes_per_sample

        self.report: Optional[VramReport] = None
        #: Filled in on exit when CUDA is available, so projections can be checked.
        self.measured_peak: Optional[int] = None

    # ------------------------------------------------------------------ context

    def __enter__(self) -> "VRAMGuard":
        gpu = None
        if self.max_vram:
            capacity = hardware.parse_memory(self.max_vram)
            if capacity is None:
                raise ValueError(f"could not parse max_vram={self.max_vram!r}")
            gpu = hardware.custom_gpu(capacity, name=f"limit {self.max_vram}")
        else:
            gpu = _current_device_gpu(self.model)

        self.report = estimate(self.profile, self.config, gpu)
        if gpu is not None and self.report.band.is_failure:
            self.report.remediations = solve(self.report, limit=3)

        self._decide()
        self._reset_peak_stats()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.verify:
            self.measured_peak = self._read_peak()
        return False

    # ------------------------------------------------------------------ verdict

    def _decide(self) -> None:
        report = self.report
        if report is None or report.gpu is None:
            return

        band = report.band
        if band is RiskBand.FITS:
            return

        message = self._describe(report)
        if band is RiskBand.CERTAIN_OOM or (self.strict and band.is_failure):
            raise VramRiskError(message, report)
        if band.is_failure or band is RiskBand.TIGHT:
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _describe(self, report: VramReport) -> str:
        low, high = report.interval
        lines = [
            f"torch-guard: this configuration is projected to need "
            f"{format_bytes(report.total)} "
            f"({format_bytes(low)}-{format_bytes(high)}) on {report.gpu.name}, "
            f"which has {report.gpu.usable_gib:.1f} GiB usable. Verdict: {report.band.value}.",
            "  breakdown: "
            + ", ".join(
                f"{label} {format_bytes(value)}"
                for label, value in report.breakdown.items()
                if value
            ),
        ]
        if report.breakdown.activations == 0:
            lines.append(
                "  activations were not estimated (pass seq_len= or image_size=), so the "
                "real peak is higher than the figure above."
            )
        fitting = next((r for r in report.remediations if r.fits), None)
        if fitting is not None:
            lines.append(f"  smallest change that fits: {fitting.label}")
        return "\n".join(lines)

    # -------------------------------------------------------------------- torch

    def _reset_peak_stats(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _read_peak(self) -> Optional[int]:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            reserved = torch.cuda.max_memory_reserved()
            context = 0
            if self.report is not None and self.report.gpu is not None:
                context = getattr(self.report.gpu, "context_bytes", None) or 0
            return reserved + context
        except Exception:
            return None

    # --------------------------------------------------------------- reporting

    @property
    def accuracy(self) -> Optional[float]:
        """Relative error of the projection against the measured peak, once known."""
        if self.measured_peak is None or self.report is None or not self.report.total:
            return None
        return (self.report.total - self.measured_peak) / self.measured_peak
