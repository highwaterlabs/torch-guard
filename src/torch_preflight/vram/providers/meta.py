"""Meta-device provider: exact parameter and activation measurement.

Requires the ``[vram]`` extra. ``torch`` is imported inside the functions, so importing
this module on a machine without it is harmless — the registry probes
:func:`available` and falls through to the static provider.

How it works
------------
``torch.device("meta")`` instantiates a model with **zero bytes allocated**: every tensor
carries real shape and dtype but no storage. Parameter counting is then exact, and running
a forward pass under ``torch.autograd.graph.saved_tensors_hooks`` captures precisely the
tensors autograd retains for backward — which *is* activation memory, by definition rather
than by formula.

Two details are load-bearing, both established by spike 0001:

* **Parameter storages must be excluded.** A ``Linear`` saves its weight to compute the
  input gradient, but that weight is the same buffer as the parameter, already counted in
  the weights term. Including it roughly doubles the activation figure.
* **Dedup by ``storage._cdata``, not ``data_ptr()``.** Views share one buffer, so summing
  saved tensors naively over-counts by ~19%. Every meta tensor has ``data_ptr() == 0``, so
  the usual key is useless there; the storage object's identity works and was verified to
  give byte-identical results to ``data_ptr()`` on CPU.

Capture is forward-only, so it excludes the transient where a checkpointed layer is
recomputed during backward. The cost model adds that term analytically.

**This provider imports and executes user code.** It is never reached from
``torch-preflight check`` — only from an explicit ``estimate --model module:factory``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from typing import Any, Dict, List, Optional, Tuple

from ..types import Confidence, ModelKind, ModelProfile, RunConfig


def available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class EntryPointError(RuntimeError):
    """Raised when a ``module:factory`` reference cannot be turned into a model."""


def parse_entry_point(reference: str) -> Tuple[str, str]:
    # rpartition, so a Windows drive letter in a file path does not split wrongly.
    module_path, _, attribute = reference.rpartition(":")
    if not module_path or not attribute:
        raise EntryPointError(
            f"{reference!r} is not a valid entry point. Expected 'module.path:factory', "
            f"e.g. 'mypkg.models:build_gpt'."
        )
    return module_path, attribute


def parse_model_args(pairs: Optional[List[str]]) -> Dict[str, Any]:
    """Turn ``["num_classes=10", "width=1.5", "pretrained=false"]`` into kwargs."""
    kwargs: Dict[str, Any] = {}
    for pair in pairs or []:
        key, separator, raw = pair.partition("=")
        if not separator:
            raise EntryPointError(f"--model-args expects key=value, got {pair!r}")
        kwargs[key.strip()] = _coerce_literal(raw.strip())
    return kwargs


def _coerce_literal(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            pass
    return raw


def build_model(reference: str, kwargs: Optional[Dict[str, Any]] = None):
    """Import ``module:factory`` and call it under the meta device.

    The factory may be a class or a function; either way the result must be an
    ``nn.Module``. Nothing is allocated — meta tensors have no storage.
    """
    import torch
    import torch.nn as nn

    module_path, attribute = parse_entry_point(reference)

    try:
        module = _import_module(module_path)
    except EntryPointError:
        raise
    except Exception as exc:
        raise EntryPointError(
            f"could not import {module_path!r}: {type(exc).__name__}: {exc}"
        ) from exc

    factory = getattr(module, attribute, None)
    if factory is None:
        raise EntryPointError(f"{module_path!r} has no attribute {attribute!r}")
    if not callable(factory):
        raise EntryPointError(f"{reference} is not callable")

    try:
        with torch.device("meta"):
            model = factory(**(kwargs or {}))
    except TypeError as exc:
        raise EntryPointError(
            f"could not construct {reference}: {exc}. Pass constructor arguments with "
            f"--model-args key=value."
        ) from exc
    except Exception as exc:
        raise EntryPointError(
            f"{reference} raised {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(model, nn.Module):
        raise EntryPointError(
            f"{reference} returned {type(model).__name__}, expected an nn.Module"
        )
    return model


def _import_module(module_path: str):
    """Import by dotted path, or by file location when given a ``.py`` path.

    The file form exists for autodetection, which resolves a class defined in the script
    being analysed. Callers are responsible for checking that the file is safe to import
    — see ``autodetect.module_is_import_safe``.
    """
    if module_path.endswith(".py"):
        if not os.path.isfile(module_path):
            raise EntryPointError(f"no such file: {module_path}")
        name = "_torch_preflight_target_" + os.path.basename(module_path)[:-3]
        spec = importlib.util.spec_from_file_location(name, module_path)
        if spec is None or spec.loader is None:
            raise EntryPointError(f"could not load {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return importlib.import_module(module_path)


def measure_parameters(model) -> Tuple[int, int, int]:
    """Return ``(param_count, trainable_count, buffer_bytes)``.

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

    return total, trainable, buffer_bytes


def _candidate_inputs(config: RunConfig) -> List[Tuple[tuple, dict]]:
    """Plausible forward inputs, most likely first.

    We do not know the model's signature, so we try the shapes implied by the run config
    and keep the first that produces a forward pass.
    """
    import torch

    batch = max(config.batch_size, 1)
    candidates: List[Tuple[tuple, dict]] = []

    if config.seq_len:
        ids = torch.zeros(batch, config.seq_len, dtype=torch.long, device="meta")
        candidates.append(((ids,), {}))
        candidates.append(((), {"input_ids": ids}))

    if config.image_size:
        pixels = torch.zeros(
            batch, 3, config.image_size, config.image_size, device="meta"
        )
        candidates.append(((pixels,), {}))
        candidates.append(((), {"pixel_values": pixels}))

    return candidates


def measure_activations(model, config: RunConfig) -> Optional[int]:
    """Bytes autograd would retain for backward, or None if no input shape worked."""
    import torch
    from torch.autograd.graph import saved_tensors_hooks

    candidates = _candidate_inputs(config)
    if not candidates:
        return None

    excluded = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        try:
            excluded.add(tensor.untyped_storage()._cdata)
        except Exception:
            continue

    model.train()  # dropout is only active in training, and it retains a mask

    for args, kwargs in candidates:
        storages: Dict[int, int] = {}

        def pack(tensor):
            try:
                storage = tensor.untyped_storage()
            except Exception:
                return tensor
            if storage._cdata not in excluded:
                storages[storage._cdata] = storage.nbytes()
            return tensor

        try:
            with torch.enable_grad():
                with saved_tensors_hooks(pack, lambda t: t):
                    model(*args, **kwargs)
        except Exception:
            continue  # wrong signature or unsupported op on meta; try the next shape

        if storages:
            return sum(storages.values())

    return None


def profile(
    reference: str,
    config: Optional[RunConfig] = None,
    model_args: Optional[Dict[str, Any]] = None,
) -> ModelProfile:
    """Build a :class:`ModelProfile` by instantiating the model on the meta device."""
    config = config or RunConfig()
    model = build_model(reference, model_args)

    total, trainable, buffer_bytes = measure_parameters(model)
    activation_bytes = measure_activations(model, config)

    per_sample = None
    if activation_bytes is not None:
        # Meta models are fp32, so the measurement is in 4-byte activations. The cost
        # model expresses activations against a 2-byte reference and rescales by the
        # configured precision, so normalise here or AMP would not shrink the term.
        per_sample = int(activation_bytes / max(config.batch_size, 1) / 2)

    return ModelProfile(
        name=reference,
        param_count=total,
        trainable_params=trainable,
        source="meta-device",
        confidence=Confidence.HIGH,
        kind=ModelKind.UNKNOWN,
        buffer_bytes=buffer_bytes,
        activation_bytes_per_sample=per_sample,
    )
