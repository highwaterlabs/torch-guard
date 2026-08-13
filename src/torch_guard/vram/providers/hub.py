"""Hub provider: resolve an architecture from its Hugging Face ``config.json``.

Requires the ``[hub]`` extra. **Never reached from ``torch-guard check``** — only from an
explicit ``estimate`` with network access enabled (RFC 0001 §4).

``huggingface_hub`` is imported inside the function, so importing this module on a machine
without the extra is harmless.
"""

from __future__ import annotations

from typing import Optional

from ..costmodel import params_from_transformer_shape
from ..types import Confidence, ModelKind, ModelProfile, TransformerShape

#: Architectures whose MLP is gated (three matrices: gate, up, down) rather than two.
_GATED_MODEL_TYPES = frozenset(
    {"llama", "mistral", "mixtral", "qwen2", "qwen3", "gemma", "gemma2", "phi3", "olmo",
     "starcoder2", "deepseek", "deepseek_v2", "cohere", "granite"}
)

#: Config keys differ across model families; try each in order.
_FIELD_ALIASES = {
    "layers": ("num_hidden_layers", "n_layer", "n_layers", "num_layers"),
    "hidden": ("hidden_size", "n_embd", "d_model", "dim"),
    "heads": (
        "num_attention_heads", "n_head", "n_heads", "num_heads",
        "encoder_attention_heads",
    ),
    "kv_heads": ("num_key_value_heads", "num_kv_heads"),
    # DistilBERT names these `dim` and `hidden_dim`, where `hidden_dim` is the *feed
    # forward* width — so it belongs here, not in the hidden-size list.
    "intermediate": ("intermediate_size", "n_inner", "ffn_dim", "d_ff", "hidden_dim"),
    "vocab": ("vocab_size",),
    "max_position": ("max_position_embeddings", "n_positions", "max_seq_len"),
}


#: Architectures with a learned position-embedding table rather than RoPE/sinusoidal.
_LEARNED_POSITION_TYPES = frozenset(
    {"gpt2", "bert", "roberta", "electra", "deberta", "distilbert", "albert",
     "xlm-roberta", "camembert", "gpt_neo", "gptj", "opt"}
)


def _tied_embeddings(config: dict) -> bool:
    """Whether the output head shares the embedding matrix.

    Absence means **True**: that is transformers' own default in ``PretrainedConfig``.
    Defaulting to False instead double-counts ``vocab x hidden`` — it put GPT-2 31% over
    its published parameter count.
    """
    for key in ("tie_word_embeddings", "tie_weights_"):
        value = config.get(key)
        if isinstance(value, bool):
            return value
    return True


def _uses_dropout(config: dict) -> bool:
    """True when any dropout probability is above zero. p=0.0 costs nothing."""
    keys = (
        "attention_probs_dropout_prob", "attn_pdrop", "attention_dropout",
        "hidden_dropout_prob", "resid_pdrop", "dropout", "dropout_rate",
    )
    return any(
        isinstance(config.get(key), (int, float)) and config[key] > 0 for key in keys
    )


def _pick(config: dict, field: str) -> Optional[int]:
    for key in _FIELD_ALIASES[field]:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def shape_from_config(config: dict) -> Optional[TransformerShape]:
    layers = _pick(config, "layers")
    hidden = _pick(config, "hidden")
    heads = _pick(config, "heads")
    if not (layers and hidden and heads):
        return None

    model_type = str(config.get("model_type", "")).lower()
    return TransformerShape(
        layers=layers,
        hidden=hidden,
        heads=heads,
        vocab=_pick(config, "vocab") or 0,
        intermediate=_pick(config, "intermediate") or 0,
        kv_heads=_pick(config, "kv_heads"),
        max_position=_pick(config, "max_position") or 0,
        tied_embeddings=_tied_embeddings(config),
        gated_mlp=model_type in _GATED_MODEL_TYPES,
        learned_positions=model_type in _LEARNED_POSITION_TYPES,
        uses_dropout=_uses_dropout(config),
        has_lm_head=any(
            "forcausallm" in str(a).lower().replace("_", "")
            or "lmhead" in str(a).lower().replace("_", "")
            for a in config.get("architectures", [])
        ),
    )


def fetch_config(repo_id: str) -> Optional[dict]:
    """Download ``config.json`` for a repo. Returns None if the extra is missing."""
    try:
        from huggingface_hub import hf_hub_download   # imported HERE, not at module top
    except ImportError:
        return None

    import json

    try:
        path = hf_hub_download(repo_id=repo_id, filename="config.json")
    except Exception:
        # Network failure, private repo, 404 — all mean "cannot resolve", not "crash".
        return None

    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def available() -> bool:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return False
    return True


def resolve(reference: str) -> Optional[ModelProfile]:
    config = fetch_config(reference)
    if config is None:
        return None

    shape = shape_from_config(config)
    if shape is None:
        return None

    params = params_from_transformer_shape(shape)
    return ModelProfile(
        name=reference,
        param_count=params,
        trainable_params=params,
        source="hub",
        confidence=Confidence.MEDIUM,
        kind=ModelKind.TRANSFORMER,
        shape=shape,
    )
