"""GPU and cloud-instance database.

Capacities here are **nominal**: the marketed board memory. What a process can actually
allocate is lower, because the driver, ECC and (on consumer cards) the display output all
take a cut before your first tensor. That gap is modelled by ``reserve_fraction`` and
``display_reserve_mib`` rather than baked into a single number, so the assumption stays
visible and overridable.

The CUDA context is deliberately *not* subtracted here — it is a separate term in the cost
model, because it is a property of the process rather than of the board.

Anything unlisted can be given directly: ``--gpu-memory 48GiB``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import GIB, MIB, TIB


@dataclass(frozen=True)
class Gpu:
    key: str
    name: str
    #: Marketed board memory in MiB.
    total_mib: int
    architecture: str = ""
    vendor: str = "nvidia"
    supports_bf16: bool = True
    #: Driver + ECC reserve, as a fraction of total. Datacentre cards run ECC by default,
    #: which costs more than the driver reserve alone.
    reserve_fraction: float = 0.02
    #: Memory held by an attached display. Zero for headless datacentre parts.
    display_reserve_mib: int = 0
    #: Measured CUDA context for this specific card, when we have a number for it.
    #: Falls back to ``costmodel.CUDA_CONTEXT_BYTES`` when unset.
    context_mib: Optional[int] = None
    aliases: Tuple[str, ...] = ()

    @property
    def total_bytes(self) -> int:
        return self.total_mib * MIB

    @property
    def usable_bytes(self) -> int:
        """What a training process can realistically allocate."""
        usable = self.total_mib * (1.0 - self.reserve_fraction) - self.display_reserve_mib
        return int(max(usable, 0) * MIB)

    @property
    def context_bytes(self) -> Optional[int]:
        return self.context_mib * MIB if self.context_mib else None

    @property
    def usable_gib(self) -> float:
        return self.usable_bytes / GIB


def _gpu(key, name, total_mib, arch, **kwargs) -> Gpu:
    return Gpu(key=key, name=name, total_mib=total_mib, architecture=arch, **kwargs)


# --------------------------------------------------------------------------- datacentre

_DATACENTRE = [
    _gpu("h200", "NVIDIA H200", 143360, "hopper", reserve_fraction=0.025),
    _gpu("h100-80gb", "NVIDIA H100 80GB", 81920, "hopper", reserve_fraction=0.025,
         aliases=("h100", "h100-sxm", "h100-pcie")),
    _gpu("h100-94gb", "NVIDIA H100 NVL 94GB", 96256, "hopper", reserve_fraction=0.025),
    _gpu("a100-80gb", "NVIDIA A100 80GB", 81920, "ampere", reserve_fraction=0.025,
         aliases=("a100",)),
    _gpu("a100-40gb", "NVIDIA A100 40GB", 40960, "ampere", reserve_fraction=0.025),
    _gpu("l40s", "NVIDIA L40S", 46080, "ada", reserve_fraction=0.02),
    _gpu("l4", "NVIDIA L4", 24576, "ada", reserve_fraction=0.02),
    _gpu("a10g", "NVIDIA A10G", 24576, "ampere", reserve_fraction=0.02),
    _gpu("a40", "NVIDIA A40", 49152, "ampere", reserve_fraction=0.02),
    _gpu("a6000", "NVIDIA RTX A6000", 49152, "ampere", reserve_fraction=0.02),
    _gpu("v100-32gb", "NVIDIA V100 32GB", 32768, "volta", supports_bf16=False,
         reserve_fraction=0.025, aliases=("v100",)),
    _gpu("v100-16gb", "NVIDIA V100 16GB", 16384, "volta", supports_bf16=False,
         reserve_fraction=0.025),
    # context_mib measured directly — see tests/calibration/measured_cuda.json.
    _gpu("t4", "NVIDIA T4", 16384, "turing", supports_bf16=False, reserve_fraction=0.02,
         context_mib=135),
    _gpu("mi300x", "AMD Instinct MI300X", 196608, "cdna3", vendor="amd",
         reserve_fraction=0.025),
    _gpu("mi250x", "AMD Instinct MI250X", 131072, "cdna2", vendor="amd",
         reserve_fraction=0.025),
]

# ----------------------------------------------------------------------------- consumer

_CONSUMER = [
    _gpu("rtx5090", "NVIDIA GeForce RTX 5090", 32768, "blackwell", display_reserve_mib=400),
    _gpu("rtx4090", "NVIDIA GeForce RTX 4090", 24564, "ada", display_reserve_mib=400),
    _gpu("rtx4080", "NVIDIA GeForce RTX 4080", 16376, "ada", display_reserve_mib=400),
    _gpu("rtx4070ti", "NVIDIA GeForce RTX 4070 Ti", 12282, "ada", display_reserve_mib=400),
    _gpu("rtx3090", "NVIDIA GeForce RTX 3090", 24576, "ampere", display_reserve_mib=400,
         aliases=("rtx3090ti",)),
    _gpu("rtx3080", "NVIDIA GeForce RTX 3080", 10240, "ampere", display_reserve_mib=400),
    _gpu("rtx3060", "NVIDIA GeForce RTX 3060 12GB", 12288, "ampere", display_reserve_mib=400),
    _gpu("rtx2080ti", "NVIDIA GeForce RTX 2080 Ti", 11264, "turing",
         supports_bf16=False, display_reserve_mib=400),
]

GPUS: Dict[str, Gpu] = {}
for _entry in _DATACENTRE + _CONSUMER:
    GPUS[_entry.key] = _entry


# ---------------------------------------------------------------------- cloud instances


@dataclass(frozen=True)
class Instance:
    key: str
    provider: str
    gpu_key: str
    count: int

    @property
    def gpu(self) -> Gpu:
        return GPUS[self.gpu_key]


def _inst(key, provider, gpu_key, count) -> Instance:
    return Instance(key=key, provider=provider, gpu_key=gpu_key, count=count)


_INSTANCE_LIST = [
    # AWS EC2 — SageMaker exposes the same shapes with an "ml." prefix, added below.
    _inst("p5.48xlarge", "aws", "h100-80gb", 8),
    _inst("p4de.24xlarge", "aws", "a100-80gb", 8),
    _inst("p4d.24xlarge", "aws", "a100-40gb", 8),
    _inst("p3dn.24xlarge", "aws", "v100-32gb", 8),
    _inst("p3.16xlarge", "aws", "v100-16gb", 8),
    _inst("p3.8xlarge", "aws", "v100-16gb", 4),
    _inst("p3.2xlarge", "aws", "v100-16gb", 1),
    _inst("g6e.xlarge", "aws", "l40s", 1),
    _inst("g6.xlarge", "aws", "l4", 1),
    _inst("g5.48xlarge", "aws", "a10g", 8),
    _inst("g5.xlarge", "aws", "a10g", 1),
    _inst("g4dn.12xlarge", "aws", "t4", 4),
    _inst("g4dn.xlarge", "aws", "t4", 1),
    # GCP
    _inst("a3-highgpu-8g", "gcp", "h100-80gb", 8),
    _inst("a2-ultragpu-8g", "gcp", "a100-80gb", 8),
    _inst("a2-ultragpu-1g", "gcp", "a100-80gb", 1),
    _inst("a2-highgpu-8g", "gcp", "a100-40gb", 8),
    _inst("a2-highgpu-1g", "gcp", "a100-40gb", 1),
    # Azure
    _inst("standard_nd96isr_h100_v5", "azure", "h100-80gb", 8),
    _inst("standard_nd96amsr_a100_v4", "azure", "a100-80gb", 8),
    _inst("standard_nc24ads_a100_v4", "azure", "a100-80gb", 1),
]

INSTANCES: Dict[str, Instance] = {}
for _entry in _INSTANCE_LIST:
    INSTANCES[_entry.key] = _entry
    if _entry.provider == "aws":
        # SageMaker training instance names mirror EC2 with an ml. prefix.
        INSTANCES[f"ml.{_entry.key}"] = Instance(
            key=f"ml.{_entry.key}",
            provider="sagemaker",
            gpu_key=_entry.gpu_key,
            count=_entry.count,
        )


# --------------------------------------------------------------------------- resolution

_ALIAS_INDEX: Dict[str, str] = {}
for _key, _entry in GPUS.items():
    _ALIAS_INDEX[_key] = _key
    for _alias in _entry.aliases:
        _ALIAS_INDEX[_alias] = _key


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "-")


def parse_memory(text: str) -> Optional[int]:
    """Parse ``48GiB`` / ``24gb`` / ``2TiB`` / ``40960MiB`` / a bare byte count into bytes.

    Longer suffixes are tested first so ``gib`` is not matched as ``g``.
    """
    value = text.strip().lower().replace(" ", "")
    for suffix, scale in (
        ("tib", TIB), ("tb", TIB),
        ("gib", GIB), ("gb", GIB),
        ("mib", MIB), ("mb", MIB),
        ("g", GIB),
    ):
        if value.endswith(suffix):
            try:
                return int(float(value[: -len(suffix)]) * scale)
            except ValueError:
                return None
    try:
        return int(value)
    except ValueError:
        return None


def resolve(name: str) -> Tuple[Optional[Gpu], int]:
    """Resolve a GPU key, alias or cloud instance name to ``(gpu, device_count)``."""
    key = _normalize(name)

    instance = INSTANCES.get(key)
    if instance is not None:
        return instance.gpu, instance.count

    gpu_key = _ALIAS_INDEX.get(key)
    if gpu_key is not None:
        return GPUS[gpu_key], 1

    # Tolerate "8xa100-80gb" and "a100-80gb x8".
    for separator in ("x", "*"):
        if separator in key:
            left, _, right = key.partition(separator)
            for count_part, name_part in ((left, right), (right, left)):
                if count_part.isdigit():
                    gpu, _ = resolve(name_part)
                    if gpu is not None:
                        return gpu, int(count_part)

    return None, 1


def custom_gpu(total_bytes: int, name: str = "custom") -> Gpu:
    """Build a Gpu from an explicit capacity, for hardware we do not know about."""
    return Gpu(
        key="custom",
        name=name,
        total_mib=total_bytes // MIB,
        architecture="unknown",
        reserve_fraction=0.02,
    )


def known_gpus() -> List[Gpu]:
    return sorted(GPUS.values(), key=lambda g: (-g.total_mib, g.key))


def known_instances() -> List[Instance]:
    seen = set()
    out = []
    for instance in INSTANCES.values():
        if instance.key in seen:
            continue
        seen.add(instance.key)
        out.append(instance)
    return sorted(out, key=lambda i: (i.provider, i.key))
