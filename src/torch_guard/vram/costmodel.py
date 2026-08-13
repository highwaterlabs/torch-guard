"""The memory cost model.

```
peak = weights + gradients + optimizer state + master weights
     + activations + CUDA context + fragmentation
```

Pure arithmetic, zero dependencies. Both the static and the meta-device providers feed the
same function, which is the whole point of the split in RFC 0001 §3.

**Every constant in the CALIBRATION section is an empirical value, not a derived truth.**
They come from published accounting (the Megatron-LM activation formula) and from commonly
reported allocator behaviour. They are wrong in some regimes. `tests/calibration/` exists to
measure how wrong, and any change to these numbers should move a calibration fixture.
"""

from __future__ import annotations

from typing import List, Optional

from .types import (
    MIB,
    Confidence,
    MemoryBreakdown,
    ModelKind,
    ModelProfile,
    OptimizerKind,
    PrecisionMode,
    RiskBand,
    RunConfig,
    Sharding,
    TransformerShape,
    VramReport,
)

# ============================== CALIBRATION CONSTANTS ==============================

#: Per-process CUDA context: driver state, kernels, cuBLAS/cuDNN workspaces. Exists before
#: your first tensor and is invisible to ``torch.cuda.memory_allocated()``.
#:
#: MEASURED at 135 MiB on a Tesla T4 (torch 2.11, CUDA 12.8) — 105 MiB after init, rising
#: to 131 after the first cuBLAS call and 135 after cuDNN. We previously assumed 600 MiB
#: on no evidence. A single card is a thin basis: larger GPUs plausibly carry a bigger
#: context, so :attr:`hardware.Gpu.context_mib` overrides this per device as data arrives.
CUDA_CONTEXT_BYTES = 135 * MIB

#: The caching allocator reserves more than it hands out.
#:
#: MEASURED at 0.105 mean (range 0.062-0.125) across six real models on a T4 -- GPT-2,
#: BERT and DistilBERT at two shapes each.
#:
#: A synthetic sweep of hand-built transformer stacks gave 0.059, and adopting that would
#: have been a mistake: toy stacks allocate a handful of uniform tensors, while real models
#: churn through embeddings, masks and head projections of many different sizes. Calibrate
#: allocator behaviour against real models, not against a microbenchmark.
FRAGMENTATION_FRACTION = 0.105

#: Activation accounting per transformer layer:
#:     bytes ~= s * b * h * (ACT_LINEAR_COEFF + ACT_ATTN_COEFF * a * s / h)
#:
#: MEASURED on torch 2.13 by tests/calibration/measure_activations.py, which sweeps the
#: sequence length and fits alpha*s + beta*s^2 to the bytes autograd actually retains.
#: The published Megatron-LM constants (34, 5) sit between the two regimes below.
#:
#: Dropout is the discriminator. With p>0 the attention path retains three tensors of
#: b*a*s^2 (softmax output, dropout mask, dropout output) instead of one, tripling the
#: quadratic term. Modern LLMs ship p=0.0, which short-circuits and saves nothing.
ACT_LINEAR_COEFF_DROPOUT = 36.0
ACT_LINEAR_COEFF_NO_DROPOUT = 32.0
ACT_ATTN_COEFF_DROPOUT = 6.0
ACT_ATTN_COEFF_NO_DROPOUT = 2.0

#: Backwards-compatible midpoints, used when the architecture is unknown.
ACT_LINEAR_COEFF = 34.0
ACT_ATTN_COEFF = 5.0
ACT_REFERENCE_DTYPE_BYTES = 2

#: With full per-layer checkpointing only the layer input is stored. MEASURED at exactly
#: 2.00 — one fp32 tensor of s*b*h, normalised to the 2-byte reference.
CHECKPOINT_ACT_COEFF = 2.0

#: During recompute, one layer's full activations are live at once.
CHECKPOINT_RECOMPUTE_LAYERS = 1

#: Inference keeps only a couple of layers' activations alive at a time.
INFERENCE_LIVE_LAYERS = 2

#: An LM head produces [batch, seq, vocab] logits, and the loss keeps more than one copy:
#: the head's output in the autocast dtype, an fp32 upcast (``cross_entropy`` runs in fp32),
#: and the ``log_softmax`` output retained for backward. Hence
#:     bytes = b * s * vocab * (activation_dtype + LM_HEAD_UPCAST + LM_HEAD_LOGSOFTMAX)
#:
#: This is derived from the operations involved, not fitted. Measurement shows it is a
#: FLOOR: GPT-2 at batch 8 x seq 256 still exceeds the resulting estimate by roughly 20%,
#: so real loss implementations keep temporaries this does not capture (HF's
#: ``shift_logits`` contiguous copy is one). Tracked in design/TODO.md.
LM_HEAD_UPCAST_BYTES = 4
LM_HEAD_LOGSOFTMAX_BYTES = 4

# ===================================================================================


def transformer_activation_bytes(
    shape: TransformerShape, config: RunConfig, seq_len: int
) -> int:
    """Activation memory for a transformer, in bytes, for one device's micro-batch."""
    b = config.batch_size
    s = seq_len
    h = shape.hidden
    a = shape.heads
    dtype_scale = config.precision.activation_bytes / ACT_REFERENCE_DTYPE_BYTES

    if shape.uses_dropout:
        linear_coeff, attn_coeff = ACT_LINEAR_COEFF_DROPOUT, ACT_ATTN_COEFF_DROPOUT
    else:
        linear_coeff, attn_coeff = ACT_LINEAR_COEFF_NO_DROPOUT, ACT_ATTN_COEFF_NO_DROPOUT

    # Per-layer, per-batch linear term: everything that is not the attention score matrix.
    linear = linear_coeff * s * b * h

    # The attention score matrix is the O(s^2) term. Flash attention never materialises it.
    attention = 0.0 if config.flash_attention else attn_coeff * a * s * s * b

    if config.gradient_checkpointing:
        # Only layer boundaries are stored, plus one layer live during recompute.
        stored = CHECKPOINT_ACT_COEFF * s * b * h * shape.layers
        live = (linear + attention) * CHECKPOINT_RECOMPUTE_LAYERS
        return int((stored + live) * dtype_scale)

    if config.inference_only:
        return int((linear + attention) * INFERENCE_LIVE_LAYERS * dtype_scale)

    return int((linear + attention) * shape.layers * dtype_scale)


def cnn_activation_bytes(profile: ModelProfile, config: RunConfig) -> Optional[int]:
    """Activation memory for a vision model, scaled from a reference resolution."""
    if profile.activation_bytes_per_sample is None:
        return None

    per_sample = profile.activation_bytes_per_sample
    # Feature maps scale with spatial area.
    if config.image_size and getattr(profile, "reference_image_size", None):
        reference = getattr(profile, "reference_image_size")
        per_sample = int(per_sample * (config.image_size / reference) ** 2)

    scale = config.precision.activation_bytes / ACT_REFERENCE_DTYPE_BYTES
    total = per_sample * config.batch_size * scale

    if config.gradient_checkpointing:
        total *= 0.3  # rough: only checkpoint boundaries survive
    if config.inference_only:
        total *= 0.2
    return int(total)


def lm_head_bytes(shape: TransformerShape, config: RunConfig, seq_len: int) -> int:
    """Logits and loss temporaries for a model with a vocabulary projection."""
    if not shape.has_lm_head or not shape.vocab:
        return 0

    elements = config.batch_size * seq_len * shape.vocab
    per_element = config.precision.activation_bytes
    if not config.inference_only:
        per_element += LM_HEAD_UPCAST_BYTES + LM_HEAD_LOGSOFTMAX_BYTES
    return elements * per_element


def _activation_bytes(profile: ModelProfile, config: RunConfig) -> Optional[int]:
    if profile.activation_bytes_per_sample is not None:
        return cnn_activation_bytes(profile, config)

    if profile.shape is not None:
        seq_len = config.seq_len or profile.shape.max_position
        if seq_len:
            return (
                transformer_activation_bytes(profile.shape, config, seq_len)
                + lm_head_bytes(profile.shape, config, seq_len)
            )

    return None


def estimate(
    profile: ModelProfile,
    config: RunConfig,
    gpu: Optional[object] = None,
    gpu_count: int = 1,
) -> VramReport:
    """Project peak VRAM for one device."""
    breakdown = MemoryBreakdown()
    notes: List[str] = []
    extra_uncertainty = 0.0

    if not profile.resolved:
        return VramReport(
            profile=profile,
            config=config,
            breakdown=breakdown,
            gpu=gpu,
            gpu_count=gpu_count,
            band=RiskBand.UNKNOWN,
            notes=[profile.reason or "model could not be resolved"],
        )

    params = profile.param_count
    trainable = profile.trainable_params or params
    if config.frozen_fraction > 0:
        trainable = int(params * (1.0 - config.frozen_fraction))

    precision = config.precision
    world = max(config.world_size, 1)

    # Sharding divides different terms depending on the ZeRO stage.
    param_div = world if config.sharding is Sharding.ZERO3 else 1
    grad_div = world if config.sharding in (Sharding.ZERO2, Sharding.ZERO3) else 1
    opt_div = world if config.sharding in (Sharding.ZERO1, Sharding.ZERO2, Sharding.ZERO3) else 1

    breakdown.weights = int(params * precision.param_bytes / param_div)

    if not config.inference_only:
        breakdown.gradients = int(trainable * precision.grad_bytes / grad_div)
        breakdown.optimizer_state = int(
            trainable * config.optimizer.states * config.optimizer.bytes_per_state / opt_div
        )
        breakdown.master_weights = int(
            trainable * precision.master_copy_bytes / opt_div
        )

    # autocast holds its casted weight copies through the backward pass, in inference too.
    breakdown.autocast_cache = int(params * precision.cast_cache_bytes / param_div)

    activations = _activation_bytes(profile, config)
    if activations is None:
        notes.append(
            "Activation memory could not be estimated (no sequence length or architecture "
            "dimensions available). The real peak will be higher than shown."
        )
        extra_uncertainty += 0.25
    else:
        breakdown.activations = activations

    breakdown.cuda_context = getattr(gpu, "context_bytes", None) or CUDA_CONTEXT_BYTES

    allocated = (
        breakdown.weights
        + breakdown.gradients
        + breakdown.optimizer_state
        + breakdown.master_weights
        + breakdown.autocast_cache
        + breakdown.activations
    )
    breakdown.fragmentation = int(allocated * FRAGMENTATION_FRACTION)

    notes.extend(_advisory_notes(profile, config, gpu))

    report = VramReport(
        profile=profile,
        config=config,
        breakdown=breakdown,
        gpu=gpu,
        gpu_count=gpu_count,
        notes=notes,
        extra_uncertainty=extra_uncertainty,
    )
    report.band = _band(report)
    return report


def _band(report: VramReport) -> RiskBand:
    """Bands from the error interval, never a fabricated probability (RFC 0001 §7)."""
    if report.gpu is None or not report.profile.resolved:
        return RiskBand.UNKNOWN

    usable = report.gpu.usable_bytes
    low, high = report.interval
    total = report.total

    if high < usable:
        return RiskBand.FITS
    if total < usable:
        return RiskBand.TIGHT
    if low < usable:
        return RiskBand.LIKELY_OOM
    return RiskBand.CERTAIN_OOM


def _advisory_notes(profile: ModelProfile, config: RunConfig, gpu) -> List[str]:
    notes: List[str] = []

    if config.accumulation_steps > 1:
        notes.append(
            f"Gradient accumulation ({config.accumulation_steps} steps) multiplies the "
            f"effective batch to {config.global_batch} but does not change peak memory — "
            f"activations scale with the micro-batch of {config.batch_size}."
        )

    if config.precision is PrecisionMode.AMP:
        notes.append(
            "torch.autocast keeps parameters and gradients in fp32; only activations move "
            "to low precision. Casting the model itself is what halves the weight term."
        )

    if gpu is not None and not gpu.supports_bf16 and config.precision in (
        PrecisionMode.PURE_BF16,
        PrecisionMode.AMP,
    ):
        notes.append(f"{gpu.name} has no bf16 support; fp16 will be used instead.")

    if config.sharding is Sharding.DDP and config.world_size > 1:
        notes.append(
            f"DDP keeps a full replica on every rank, so {config.world_size} GPUs do not "
            f"reduce per-device memory. FSDP/ZeRO-3 would shard it."
        )

    if (
        profile.shape is not None
        and config.seq_len
        and not config.flash_attention
        and config.seq_len >= 2048
    ):
        notes.append(
            f"The attention score matrix is O(seq²) and dominates at seq={config.seq_len}. "
            f"Flash attention / SDPA would remove that term entirely."
        )

    return notes


def params_from_transformer_shape(shape: TransformerShape) -> int:
    """Analytic parameter count for a standard transformer.

    Verified against known models: llama-2-7b (L32 H4096 I11008 V32000, gated, untied)
    gives 6.74B, and bert-base (L12 H768 I3072 V30522) gives ~109M.
    """
    h = shape.hidden
    per_layer = 0

    # Attention: q, k, v, o projections. Grouped-query attention shrinks k and v.
    kv_ratio = (shape.kv_heads or shape.heads) / shape.heads
    per_layer += h * h                    # q
    per_layer += 2 * h * h * kv_ratio     # k, v
    per_layer += h * h                    # o

    # MLP: two matrices, or three for gated variants (SwiGLU/GeGLU).
    mlp_matrices = 3 if shape.gated_mlp else 2
    per_layer += mlp_matrices * h * shape.intermediate

    # Layer norms are negligible but cheap to include.
    per_layer += 4 * h

    total = per_layer * shape.layers
    total += shape.vocab * h                      # token embeddings
    if not shape.tied_embeddings:
        total += shape.vocab * h                  # separate output head
    if shape.learned_positions and shape.max_position:
        total += shape.max_position * h           # learned position table (not RoPE)

    return int(total)
