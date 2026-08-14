"""Shared value types for VRAM estimation.

Kept dependency-free and separate from the cost model so that providers, the extractor
and the reporters can all speak the same vocabulary without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

TIB = 1024 ** 4
GIB = 1024 ** 3
MIB = 1024 ** 2


class Confidence(str, Enum):
    """How much we trust a :class:`ModelProfile`. Drives the reported error interval."""

    HIGH = "high"          # exact param count (meta device, or an exact arch-db entry)
    MEDIUM = "medium"      # analytic formula from a known config
    LOW = "low"            # inferred from partial information
    UNKNOWN = "unknown"    # could not resolve; we refuse to produce a number

    @property
    def interval(self) -> float:
        """Relative half-width of the error interval around a point estimate."""
        return _CONFIDENCE_INTERVAL[self]


_CONFIDENCE_INTERVAL = {
    Confidence.HIGH: 0.10,
    Confidence.MEDIUM: 0.20,
    Confidence.LOW: 0.35,
    Confidence.UNKNOWN: 1.0,
}


class PrecisionMode(str, Enum):
    """How weights, gradients and activations are stored during the step.

    The distinction between :attr:`AMP` and :attr:`PURE_BF16` is the one people get wrong
    most often. ``torch.autocast`` does **not** shrink your weights: parameters and
    gradients stay fp32 and only activations move to the low-precision dtype. Casting the
    model itself (``model.bfloat16()``) is what halves the weight and gradient terms.
    """

    FP32 = "fp32"                 # everything fp32
    AMP = "amp"                   # autocast: fp32 params/grads, low-precision activations
    PURE_FP16 = "pure-fp16"       # model.half(): fp16 params, grads and activations
    PURE_BF16 = "pure-bf16"       # model.bfloat16()
    FP16_MASTER = "fp16-master"   # fp16 params + fp32 master copy (DeepSpeed/Apex O2)

    @property
    def param_bytes(self) -> int:
        return _PRECISION_TABLE[self][0]

    @property
    def grad_bytes(self) -> int:
        return _PRECISION_TABLE[self][1]

    @property
    def activation_bytes(self) -> int:
        return _PRECISION_TABLE[self][2]

    @property
    def master_copy_bytes(self) -> int:
        """Extra fp32 copy of the weights held by the optimizer, if any."""
        return _PRECISION_TABLE[self][3]

    @property
    def cast_cache_bytes(self) -> int:
        """Low-precision copies of the weights that ``torch.autocast`` caches.

        autocast casts each weight once per forward and holds the result for the backward
        pass. Measured against real models this accounts for almost the entire gap that
        remained after weights, gradients, optimizer state and activations.
        """
        return _PRECISION_TABLE[self][4]


#                       param, grad, activation, master copy, autocast cast cache
_PRECISION_TABLE: Dict[PrecisionMode, Tuple[int, int, int, int, int]] = {
    PrecisionMode.FP32: (4, 4, 4, 0, 0),
    PrecisionMode.AMP: (4, 4, 2, 0, 2),
    PrecisionMode.PURE_FP16: (2, 2, 2, 0, 0),
    PrecisionMode.PURE_BF16: (2, 2, 2, 0, 0),
    PrecisionMode.FP16_MASTER: (2, 2, 2, 4, 0),
}


class OptimizerKind(str, Enum):
    """Optimizer state cost, in states-per-parameter and bytes-per-state."""

    SGD = "sgd"
    SGD_MOMENTUM = "sgd-momentum"
    ADAM = "adam"
    ADAMW = "adamw"
    ADAM_8BIT = "adamw-8bit"
    ADAFACTOR = "adafactor"
    LION = "lion"
    LION_8BIT = "lion-8bit"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"

    @property
    def states(self) -> int:
        return _OPTIMIZER_TABLE[self][0]

    @property
    def bytes_per_state(self) -> int:
        return _OPTIMIZER_TABLE[self][1]

    @property
    def label(self) -> str:
        return _OPTIMIZER_TABLE[self][2]


#                          states, bytes/state, label
_OPTIMIZER_TABLE: Dict[OptimizerKind, Tuple[float, int, str]] = {
    OptimizerKind.SGD: (0, 4, "SGD"),
    OptimizerKind.SGD_MOMENTUM: (1, 4, "SGD + momentum"),
    OptimizerKind.ADAM: (2, 4, "Adam"),
    OptimizerKind.ADAMW: (2, 4, "AdamW"),
    OptimizerKind.ADAM_8BIT: (2, 1, "8-bit AdamW"),
    # Adafactor keeps factored second moments: O(n+m) per matrix rather than O(n*m).
    # Treated as a small constant fraction of one full state.
    OptimizerKind.ADAFACTOR: (0.1, 4, "Adafactor"),
    OptimizerKind.LION: (1, 4, "Lion"),
    OptimizerKind.LION_8BIT: (1, 1, "8-bit Lion"),
    OptimizerKind.RMSPROP: (1, 4, "RMSprop"),
    OptimizerKind.ADAGRAD: (1, 4, "Adagrad"),
}


class Sharding(str, Enum):
    """How state is split across ranks."""

    NONE = "none"
    DDP = "ddp"            # full replica per rank
    ZERO1 = "zero-1"       # optimizer state sharded
    ZERO2 = "zero-2"       # + gradients sharded
    ZERO3 = "zero-3"       # + parameters sharded (equivalently FSDP FULL_SHARD)

    @property
    def label(self) -> str:
        return {
            Sharding.NONE: "single device",
            Sharding.DDP: "DDP",
            Sharding.ZERO1: "ZeRO-1",
            Sharding.ZERO2: "ZeRO-2",
            Sharding.ZERO3: "FSDP / ZeRO-3",
        }[self]


class ModelKind(str, Enum):
    TRANSFORMER = "transformer"
    CNN = "cnn"
    ENCODER_DECODER = "encoder-decoder"
    UNKNOWN = "unknown"


@dataclass
class TransformerShape:
    """Architectural dimensions needed by the activation formula."""

    layers: int
    hidden: int
    heads: int
    vocab: int = 0
    intermediate: int = 0
    kv_heads: Optional[int] = None      # < heads means grouped-query attention
    max_position: int = 0
    tied_embeddings: bool = False
    gated_mlp: bool = False             # SwiGLU/GeGLU use three matrices, not two
    #: BERT/GPT-2 learn a position embedding table; RoPE models (Llama, Mistral) do not.
    learned_positions: bool = False
    #: Causal/masked LM heads project to the vocabulary, producing a [batch, seq, vocab]
    #: logits tensor. At large vocabularies this rivals the whole transformer stack.
    has_lm_head: bool = False
    #: Dropout with p>0 during training. It triples the quadratic attention term, because
    #: the mask and the dropout output are both retained alongside the softmax output.
    #: Modern LLMs (Llama, Mistral, Qwen) ship with p=0.0, which short-circuits entirely.
    uses_dropout: bool = False
    #: Decoder layer count for encoder-decoder models (T5, Whisper). When set, ``layers``
    #: is the *encoder* depth and a decoder layer additionally carries cross-attention.
    decoder_layers: int = 0
    #: A fixed encoder length, where the architecture imposes one. Whisper always sees
    #: 3000 mel frames, halved to 1500 positions by its stride-2 conv frontend, whatever
    #: the audio actually contains — so the encoder cost does not vary with the run.
    encoder_seq_len: Optional[int] = None
    #: Which measured coefficient set to use; see ENCODER_DECODER_COEFFS.
    activation_family: Optional[str] = None

    @property
    def is_encoder_decoder(self) -> bool:
        return self.decoder_layers > 0

    def __post_init__(self) -> None:
        if not self.intermediate:
            self.intermediate = 4 * self.hidden
        if self.kv_heads is None:
            self.kv_heads = self.heads


@dataclass
class ModelProfile:
    """What the cost model needs to know about the model itself."""

    name: str
    param_count: int
    trainable_params: int
    source: str                     # "arch-snapshot" | "formula" | "hub" | "meta-device"
    confidence: Confidence
    kind: ModelKind = ModelKind.UNKNOWN
    buffer_bytes: int = 0
    #: Set by the meta provider, or derived per-sample for CNNs in the snapshot.
    activation_bytes_per_sample: Optional[int] = None
    shape: Optional[TransformerShape] = None
    #: Why we could not resolve, when confidence is UNKNOWN.
    reason: Optional[str] = None

    @classmethod
    def unknown(cls, name: str, reason: str) -> "ModelProfile":
        return cls(
            name=name,
            param_count=0,
            trainable_params=0,
            source="unresolved",
            confidence=Confidence.UNKNOWN,
            reason=reason,
        )

    @property
    def resolved(self) -> bool:
        return self.confidence is not Confidence.UNKNOWN


@dataclass
class RunConfig:
    """The training configuration, as extracted from the script or given on the CLI."""

    #: Per-device micro-batch. This is what activation memory scales with — not the
    #: global batch, which accumulation and world size multiply up from here.
    batch_size: int = 1
    seq_len: Optional[int] = None
    image_size: Optional[int] = None
    precision: PrecisionMode = PrecisionMode.FP32
    optimizer: OptimizerKind = OptimizerKind.ADAMW
    gradient_checkpointing: bool = False
    flash_attention: bool = False
    accumulation_steps: int = 1
    world_size: int = 1
    sharding: Sharding = Sharding.NONE
    #: Autoregressive decoding, which caches K and V for every token generated so far.
    #: Distinct from ``inference_only``: a forward pass over a batch builds no cache.
    generation: bool = False
    #: Total context the cache has to hold — prompt plus everything generated. Defaults to
    #: ``seq_len`` when not given.
    max_context: Optional[int] = None
    #: DeepSpeed ZeRO-Offload: optimizer state and the fp32 master copy live in CPU memory
    #: rather than on the device. Read from ``zero_optimization.offload_optimizer``.
    offload_optimizer: bool = False
    #: ``zero_optimization.offload_param``. Recorded but deliberately not subtracted from
    #: the weights term -- see the note in ``costmodel``.
    offload_params: bool = False
    #: Fraction of parameters frozen (LoRA, frozen backbone) — no grad or optimizer state.
    frozen_fraction: float = 0.0
    inference_only: bool = False

    #: Where each field came from, for "batch_size=64 (train.py:27)" reporting.
    sources: Dict[str, str] = field(default_factory=dict)

    @property
    def global_batch(self) -> int:
        return self.batch_size * self.accumulation_steps * max(self.world_size, 1)

    def replace(self, **changes) -> "RunConfig":
        from dataclasses import replace as _replace

        return _replace(self, **changes)


class RiskBand(str, Enum):
    """Verdict bands. Deliberately not a fabricated probability — see RFC 0001 §7."""

    FITS = "FITS"
    TIGHT = "TIGHT"
    LIKELY_OOM = "LIKELY_OOM"
    CERTAIN_OOM = "CERTAIN_OOM"
    UNKNOWN = "UNKNOWN"

    @property
    def is_failure(self) -> bool:
        return self in (RiskBand.LIKELY_OOM, RiskBand.CERTAIN_OOM)


@dataclass
class MemoryBreakdown:
    """Every term in bytes. Sums to :attr:`total`."""

    weights: int = 0
    gradients: int = 0
    optimizer_state: int = 0
    master_weights: int = 0
    autocast_cache: int = 0
    activations: int = 0
    #: Keys and values cached across decoding steps. Generation only — a training step or a
    #: plain forward pass builds no cache.
    kv_cache: int = 0
    cuda_context: int = 0
    fragmentation: int = 0

    @property
    def total(self) -> int:
        return (
            self.weights
            + self.gradients
            + self.optimizer_state
            + self.master_weights
            + self.autocast_cache
            + self.activations
            + self.kv_cache
            + self.cuda_context
            + self.fragmentation
        )

    def items(self) -> List[Tuple[str, int]]:
        return [
            ("weights", self.weights),
            ("gradients", self.gradients),
            ("optimizer state", self.optimizer_state),
            ("master weights", self.master_weights),
            ("autocast cache", self.autocast_cache),
            ("activations", self.activations),
            ("KV cache", self.kv_cache),
            ("CUDA context", self.cuda_context),
            ("fragmentation", self.fragmentation),
        ]


@dataclass
class Remediation:
    """A change that would reduce peak memory, and what it would cost you."""

    label: str
    saved_bytes: int
    new_total: int
    fits: bool
    #: 0 = free, higher = more disruptive to the training recipe.
    disruption: int = 0
    note: str = ""


@dataclass
class VramReport:
    profile: ModelProfile
    config: RunConfig
    breakdown: MemoryBreakdown
    gpu: Optional[object] = None      # hardware.Gpu; untyped to avoid an import cycle
    gpu_count: int = 1
    band: RiskBand = RiskBand.UNKNOWN
    remediations: List[Remediation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    #: Widens the interval when a term could not be estimated (e.g. unknown activations).
    extra_uncertainty: float = 0.0

    @property
    def total(self) -> int:
        return self.breakdown.total

    @property
    def interval(self) -> Tuple[int, int]:
        spread = min(self.profile.confidence.interval + self.extra_uncertainty, 1.0)
        return int(self.total * (1 - spread)), int(self.total * (1 + spread))

    @property
    def utilization(self) -> Optional[float]:
        if self.gpu is None:
            return None
        return self.total / self.gpu.usable_bytes


def gib(value: int) -> float:
    return value / GIB


def format_bytes(value: int) -> str:
    if value >= GIB:
        return f"{value / GIB:.2f} GiB"
    if value >= MIB:
        return f"{value / MIB:.0f} MiB"
    return f"{value} B"
