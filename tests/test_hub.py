"""Hugging Face hub provider: config field mapping, and integration against real configs.

Needs only the ``[hub]`` extra — deliberately no torch, so these run in a lightweight CI
job. The offline tests replay `config.json` files captured from the hub; the
``network``-marked ones hit it live and are deselected by default.
"""

import json
import pathlib

import pytest

from torch_preflight.vram import archdb
from torch_preflight.vram.providers import hub as hub_provider
from torch_preflight.vram.providers import resolve_profile
from torch_preflight.vram.types import Confidence, PrecisionMode, RiskBand, RunConfig



def test_hub_maps_a_llama_style_config():
    shape = hub_provider.shape_from_config({
        "model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096,
        "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336,
        "vocab_size": 128256, "max_position_embeddings": 8192,
        "tie_word_embeddings": False, "attention_dropout": 0.0,
        "architectures": ["LlamaForCausalLM"],
    })
    assert (shape.layers, shape.hidden, shape.kv_heads) == (32, 4096, 8)
    assert shape.gated_mlp and not shape.uses_dropout and not shape.learned_positions


def test_hub_maps_a_gpt2_style_config():
    shape = hub_provider.shape_from_config({
        "model_type": "gpt2", "n_layer": 12, "n_embd": 768, "n_head": 12,
        "vocab_size": 50257, "n_positions": 1024, "attn_pdrop": 0.1,
    })
    assert (shape.layers, shape.hidden, shape.heads) == (12, 768, 768 // 64)
    assert shape.uses_dropout and shape.learned_positions and not shape.gated_mlp


def test_hub_returns_none_for_an_unusable_config():
    assert hub_provider.shape_from_config({"model_type": "mystery"}) is None


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"attention_dropout": 0.0}, False),
        ({"attention_dropout": 0.1}, True),
        ({"hidden_dropout_prob": 0.1}, True),
        ({}, False),
    ],
)
def test_hub_detects_active_dropout(config, expected):
    """p=0.0 short-circuits in PyTorch and costs nothing, so it must not count."""
    assert hub_provider._uses_dropout(config) is expected



HUB_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "hub"

#: Published parameter counts for the captured configs. Two of these are deliberately
#: NOT in the bundled snapshot, so they exercise the hub path rather than arch-db.
HUB_CASES = [
    ("gpt2", 124_439_808),
    ("distilbert-base-uncased", 66_362_880),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1_100_048_384),
    ("Qwen/Qwen2-0.5B", 494_032_768),
]


@pytest.fixture
def offline_hub(monkeypatch):
    """Replay real `config.json` files captured from the hub, with no network.

    The fixtures are genuine downloads, not hand-written — which matters: the offline
    unit tests passed against configs I authored to match my own field mapping, while
    these caught a 31% parameter error on GPT-2 and a total failure on DistilBERT.
    """
    huggingface_hub = pytest.importorskip("huggingface_hub")

    def fake_download(repo_id, filename, **kwargs):
        path = HUB_FIXTURES / (repo_id.replace("/", "__") + ".json")
        if not path.exists():
            raise FileNotFoundError(f"no captured config for {repo_id}")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return fake_download


@pytest.mark.parametrize("repo,published", HUB_CASES, ids=[c[0] for c in HUB_CASES])
def test_hub_derives_published_parameter_counts(offline_hub, repo, published):
    """The whole point of the hub path: a model we have never seen, sized correctly."""
    profile = hub_provider.resolve(repo)
    assert profile is not None, f"{repo} produced no profile"
    error = abs(profile.param_count - published) / published
    assert error < 0.03, (
        f"{repo}: derived {profile.param_count:,}, published {published:,} ({error:.1%})"
    )
    assert profile.source == "hub"
    assert profile.confidence is Confidence.MEDIUM


def test_hub_defaults_tied_embeddings_to_true(offline_hub):
    """transformers' own default. Getting this wrong double-counts vocab x hidden."""
    config = json.loads((HUB_FIXTURES / "gpt2.json").read_text())
    assert "tie_word_embeddings" not in config      # absent, so the default decides
    assert hub_provider.shape_from_config(config).tied_embeddings is True


def test_hub_reads_distilberts_own_field_names(offline_hub):
    """DistilBERT uses dim/hidden_dim/n_heads/tie_weights_ and nothing else's names."""
    shape = hub_provider.shape_from_config(
        json.loads((HUB_FIXTURES / "distilbert-base-uncased.json").read_text())
    )
    assert (shape.layers, shape.hidden, shape.heads) == (6, 768, 12)
    assert shape.intermediate == 3072      # `hidden_dim` is the FFN width, not the model's
    assert shape.tied_embeddings is True   # via `tie_weights_`


def test_hub_detects_grouped_query_attention(offline_hub):
    shape = hub_provider.shape_from_config(
        json.loads((HUB_FIXTURES / "Qwen__Qwen2-0.5B.json").read_text())
    )
    assert shape.kv_heads is not None and shape.kv_heads < shape.heads


def test_resolve_profile_falls_through_to_the_hub(offline_hub):
    """Not in the snapshot, so this can only have come from the hub."""
    assert archdb.resolve("TinyLlama/TinyLlama-1.1B-Chat-v1.0") is None
    profile = resolve_profile("TinyLlama/TinyLlama-1.1B-Chat-v1.0", allow_network=True)
    assert profile.source == "hub"
    assert profile.param_count == pytest.approx(1_100_048_384, rel=0.03)


def test_snapshot_wins_over_the_hub(offline_hub):
    """A bundled entry is exact and offline, so it must not be second-guessed."""
    profile = resolve_profile("gpt2", allow_network=True)
    assert profile.source == "arch-snapshot"
    assert profile.confidence is Confidence.HIGH


def test_hub_miss_degrades_to_unknown(offline_hub):
    profile = resolve_profile("nonexistent-org/nonexistent-model", allow_network=True)
    assert profile.confidence is Confidence.UNKNOWN
    assert "could not be resolved" in profile.reason


def test_hub_is_never_consulted_without_permission(monkeypatch):
    """`check` and offline `estimate` must not touch the network, ever."""
    huggingface_hub = pytest.importorskip("huggingface_hub")

    def explode(*args, **kwargs):
        raise AssertionError("the hub was contacted without allow_network")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", explode)
    profile = resolve_profile("TinyLlama/TinyLlama-1.1B-Chat-v1.0", allow_network=False)
    assert profile.confidence is Confidence.UNKNOWN
    assert "--online" in profile.reason


def test_hub_estimate_end_to_end(offline_hub):
    from torch_preflight.vram import estimate_config

    profile = resolve_profile("TinyLlama/TinyLlama-1.1B-Chat-v1.0", allow_network=True)
    report = estimate_config(
        profile,
        RunConfig(batch_size=4, seq_len=2048, precision=PrecisionMode.AMP),
        gpu="rtx4090",
    )
    assert report.band is not RiskBand.UNKNOWN
    assert report.breakdown.activations > 0


# ------------------------------------------------------ live hub (opt-in, networked)


@pytest.mark.network
@pytest.mark.parametrize("repo,published", HUB_CASES, ids=[c[0] for c in HUB_CASES])
def test_hub_live_fetch(repo, published):
    """Hits the real hub. Deselected by default: `-m network` to run it.

    The offline tests replay captured copies of these same configs; this one catches the
    day a field gets renamed upstream.
    """
    pytest.importorskip("huggingface_hub")
    profile = hub_provider.resolve(repo)
    assert profile is not None
    assert abs(profile.param_count - published) / published < 0.03
