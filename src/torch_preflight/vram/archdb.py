"""Bundled architecture snapshot.

Ships in the base wheel, so the common case — a `from_pretrained("...")` string literal —
resolves with no network, no torch and no user code executed (RFC 0001 §4, §5 layer 1).

Anything not in the snapshot falls through to the `[hub]` provider, and failing that to an
honest ``UNKNOWN``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from .types import Confidence, ModelKind, ModelProfile, TransformerShape

_DATA_FILE = Path(__file__).parent / "data" / "architectures.json"


@lru_cache(maxsize=1)
def _load() -> Dict[str, dict]:
    with _DATA_FILE.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["models"]


@lru_cache(maxsize=1)
def _index() -> Dict[str, str]:
    """Map every key and alias (normalized) to its canonical key."""
    index: Dict[str, str] = {}
    for key, entry in _load().items():
        index[_normalize(key)] = key
        for alias in entry.get("aliases", []):
            index[_normalize(alias)] = key
    return index


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "")


def _candidates(name: str) -> List[str]:
    """Progressively looser forms of a model reference.

    ``meta-llama/Llama-2-7b-hf`` -> full name, then the basename, then without the
    ``-hf`` suffix, which is how most hub repos differ from their common name.
    """
    normalized = _normalize(name)
    forms = [normalized]

    if "/" in normalized:
        forms.append(normalized.rsplit("/", 1)[1])

    for form in list(forms):
        for suffix in ("-hf", "-v0.1", "-instruct", "-chat"):
            if form.endswith(suffix):
                forms.append(form[: -len(suffix)])

    seen = set()
    return [f for f in forms if not (f in seen or seen.add(f))]


def lookup(name: str) -> Optional[dict]:
    index = _index()
    for candidate in _candidates(name):
        key = index.get(candidate)
        if key is not None:
            entry = dict(_load()[key])
            entry["key"] = key
            return entry
    return None


def shape_from_dict(data: dict) -> TransformerShape:
    return TransformerShape(
        layers=data["layers"],
        hidden=data["hidden"],
        heads=data["heads"],
        vocab=data.get("vocab", 0),
        intermediate=data.get("intermediate", 0),
        kv_heads=data.get("kv_heads"),
        max_position=data.get("max_position", 0),
        tied_embeddings=data.get("tied_embeddings", False),
        gated_mlp=data.get("gated_mlp", False),
        learned_positions=data.get("learned_positions", False),
        uses_dropout=data.get("uses_dropout", False),
        has_lm_head=data.get("has_lm_head", False),
    )


def profile_from_entry(entry: dict) -> ModelProfile:
    shape = shape_from_dict(entry["shape"]) if "shape" in entry else None
    kind = ModelKind(entry.get("kind", "unknown"))

    profile = ModelProfile(
        name=entry["key"],
        param_count=entry["params"],
        trainable_params=entry["params"],
        source="arch-snapshot",
        confidence=Confidence.HIGH,
        kind=kind,
        shape=shape,
        activation_bytes_per_sample=entry.get("activation_bytes_per_sample"),
    )
    # Carried for the CNN activation scaler; harmless when absent.
    if "reference_image_size" in entry:
        setattr(profile, "reference_image_size", entry["reference_image_size"])
    return profile


def resolve(name: str) -> Optional[ModelProfile]:
    entry = lookup(name)
    return profile_from_entry(entry) if entry is not None else None


def known_models() -> List[str]:
    return sorted(_load())
