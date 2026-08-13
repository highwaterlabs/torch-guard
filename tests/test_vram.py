"""VRAM estimation: hardware DB, cost model, extraction, providers, solver."""

import json
import pathlib
import subprocess
import sys
import textwrap
import warnings

import pytest

from torch_preflight.vram import archdb, hardware
from torch_preflight.vram.costmodel import (
    estimate,
    params_from_transformer_shape,
    transformer_activation_bytes,
)
from torch_preflight.vram.extract import extract_from_source
from torch_preflight.vram.providers import resolve_profile, static
from torch_preflight.vram.solver import MAX_STACK, solve
from torch_preflight.vram.types import (
    GIB,
    Confidence,
    OptimizerKind,
    PrecisionMode,
    RiskBand,
    RunConfig,
    Sharding,
    TransformerShape,
)

SEVEN_B = 6_738_415_616


# ------------------------------------------------------------------- packaging


def test_base_install_never_imports_torch():
    """The invariant that keeps the default install light (RFC 0001 §4).

    One careless top-level ``import torch`` would silently make `pip install torch-preflight`
    pull in gigabytes. Normal tests cannot catch it, so this asserts it directly.
    """
    script = (
        "import sys, torch_preflight;"
        "from torch_preflight import vram;"
        "torch_preflight.check_source('t.py', 'x = 1');"
        "vram.estimate_script('t.py', 'x = 1');"
        "assert 'torch' not in sys.modules, 'base install pulled in torch';"
        "assert 'huggingface_hub' not in sys.modules, 'base install pulled in hub'"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


# -------------------------------------------------------------------- hardware


def test_gpu_resolution_by_key_and_alias():
    gpu, count = hardware.resolve("a100-80gb")
    assert gpu.key == "a100-80gb" and count == 1
    assert hardware.resolve("a100")[0].key == "a100-80gb"
    assert hardware.resolve("H100")[0].key == "h100-80gb"


def test_cloud_instance_resolution():
    gpu, count = hardware.resolve("p4de.24xlarge")
    assert gpu.key == "a100-80gb" and count == 8
    # SageMaker mirrors EC2 shapes with an ml. prefix.
    assert hardware.resolve("ml.p4d.24xlarge") == (hardware.GPUS["a100-40gb"], 8)


def test_multiplier_syntax():
    gpu, count = hardware.resolve("8xa100-80gb")
    assert gpu.key == "a100-80gb" and count == 8


def test_unknown_gpu_resolves_to_none():
    assert hardware.resolve("rtx9090") == (None, 1)


def test_usable_memory_is_below_advertised():
    """A 24GB card does not give you 24GiB — driver, ECC and display take a cut."""
    gpu = hardware.GPUS["rtx4090"]
    assert gpu.usable_bytes < gpu.total_bytes
    assert 22.0 < gpu.usable_gib < 24.0


def test_parse_memory():
    assert hardware.parse_memory("48GiB") == 48 * GIB
    assert hardware.parse_memory("24gb") == 24 * GIB
    assert hardware.parse_memory("nonsense") is None


# --------------------------------------------------------------------- archdb


@pytest.mark.parametrize(
    "reference",
    ["llama-2-7b", "meta-llama/Llama-2-7b-hf", "LLAMA-2-7B", "llama_2_7b"],
)
def test_archdb_normalizes_references(reference):
    profile = archdb.resolve(reference)
    assert profile is not None and profile.param_count == SEVEN_B


def test_archdb_miss_returns_none():
    assert archdb.resolve("definitely-not-a-model") is None


def test_archdb_entries_are_self_consistent():
    """Every entry with a shape must have its published count match the formula."""
    for name in archdb.known_models():
        profile = archdb.resolve(name)
        if profile.shape is None:
            continue
        computed = params_from_transformer_shape(profile.shape)
        error = abs(computed - profile.param_count) / profile.param_count
        assert error < 0.03, f"{name}: formula {computed} vs published {profile.param_count}"


# ----------------------------------------------------------------- cost model


def _profile(params=SEVEN_B):
    return static.from_param_count(params, name="test")


def test_fp32_adamw_is_sixteen_bytes_per_param():
    """The standard accounting: 4 weights + 4 grads + 8 optimizer."""
    report = estimate(_profile(), RunConfig(precision=PrecisionMode.FP32))
    per_param = (
        report.breakdown.weights
        + report.breakdown.gradients
        + report.breakdown.optimizer_state
    ) / SEVEN_B
    assert per_param == pytest.approx(16.0)


def test_amp_keeps_weights_fp32():
    """autocast shrinks activations, not parameters — the most common misconception."""
    fp32 = estimate(_profile(), RunConfig(precision=PrecisionMode.FP32))
    amp = estimate(_profile(), RunConfig(precision=PrecisionMode.AMP))
    assert amp.breakdown.weights == fp32.breakdown.weights
    assert amp.breakdown.gradients == fp32.breakdown.gradients


def test_pure_bf16_halves_weights():
    fp32 = estimate(_profile(), RunConfig(precision=PrecisionMode.FP32))
    bf16 = estimate(_profile(), RunConfig(precision=PrecisionMode.PURE_BF16))
    assert bf16.breakdown.weights == fp32.breakdown.weights // 2


def test_optimizer_state_scales_with_kind():
    def state(kind):
        return estimate(_profile(), RunConfig(optimizer=kind)).breakdown.optimizer_state

    assert state(OptimizerKind.SGD) == 0
    assert state(OptimizerKind.SGD_MOMENTUM) == SEVEN_B * 4
    assert state(OptimizerKind.ADAMW) == SEVEN_B * 8
    assert state(OptimizerKind.ADAM_8BIT) == SEVEN_B * 2


def test_inference_has_no_gradients_or_optimizer():
    report = estimate(_profile(), RunConfig(inference_only=True))
    assert report.breakdown.gradients == 0
    assert report.breakdown.optimizer_state == 0


def test_frozen_fraction_removes_grad_and_optimizer():
    full = estimate(_profile(), RunConfig())
    lora = estimate(_profile(), RunConfig(frozen_fraction=0.99))
    assert lora.breakdown.gradients < full.breakdown.gradients * 0.02
    assert lora.breakdown.weights == full.breakdown.weights  # weights still resident


@pytest.mark.parametrize(
    "sharding,shards_weights,shards_grads,shards_optimizer",
    [
        (Sharding.DDP, False, False, False),
        (Sharding.ZERO1, False, False, True),
        (Sharding.ZERO2, False, True, True),
        (Sharding.ZERO3, True, True, True),
    ],
)
def test_sharding_divides_the_right_terms(
    sharding, shards_weights, shards_grads, shards_optimizer
):
    single = estimate(_profile(), RunConfig())
    sharded = estimate(_profile(), RunConfig(sharding=sharding, world_size=4))

    assert (sharded.breakdown.weights < single.breakdown.weights) is shards_weights
    assert (sharded.breakdown.gradients < single.breakdown.gradients) is shards_grads
    assert (
        sharded.breakdown.optimizer_state < single.breakdown.optimizer_state
    ) is shards_optimizer


# --------------------------------------------------------------- activations


def _shape():
    return TransformerShape(layers=32, hidden=4096, heads=32, intermediate=11008)


def test_flash_attention_removes_the_quadratic_term():
    """With flash the growth is exactly linear in seq; without it, super-linear.

    Asserted as a property rather than a ratio, so it stays valid when the measured
    coefficients change.
    """
    flash = RunConfig(batch_size=4, precision=PrecisionMode.AMP, flash_attention=True)
    plain = flash.replace(flash_attention=False)

    assert transformer_activation_bytes(_shape(), flash, 2048) == pytest.approx(
        2 * transformer_activation_bytes(_shape(), flash, 1024), rel=0.01
    )
    assert transformer_activation_bytes(_shape(), plain, 2048) > 2 * (
        transformer_activation_bytes(_shape(), plain, 1024)
    )
    assert transformer_activation_bytes(_shape(), flash, 2048) < (
        transformer_activation_bytes(_shape(), plain, 2048)
    )


def test_dropout_triples_the_quadratic_attention_term():
    """Measured on torch 2.13: dropout retains the mask and its output as well as the
    softmax output, so the O(seq²) term is 3x that of a dropout-free model."""
    config = RunConfig(batch_size=2, precision=PrecisionMode.AMP)
    plain = TransformerShape(layers=8, hidden=1024, heads=16, intermediate=4096)
    with_dropout = TransformerShape(
        layers=8, hidden=1024, heads=16, intermediate=4096, uses_dropout=True
    )

    # Isolate the quadratic term by differencing two sequence lengths.
    def quadratic(shape):
        long = transformer_activation_bytes(shape, config, 4096)
        short = transformer_activation_bytes(shape, config, 2048)
        return long - 2 * short  # the linear part cancels

    assert quadratic(with_dropout) == pytest.approx(3 * quadratic(plain), rel=0.01)


def test_modern_llms_are_not_charged_for_dropout():
    """Llama and friends ship p=0.0, which short-circuits and saves nothing."""
    profile = archdb.resolve("llama-2-7b")
    assert profile.shape.uses_dropout is False
    assert archdb.resolve("bert-base-uncased").shape.uses_dropout is True


def test_activations_scale_linearly_with_batch():
    config = RunConfig(batch_size=1, seq_len=1024, precision=PrecisionMode.AMP)
    one = transformer_activation_bytes(_shape(), config, 1024)
    eight = transformer_activation_bytes(_shape(), config.replace(batch_size=8), 1024)
    assert eight == pytest.approx(one * 8, rel=0.01)


def test_attention_term_is_quadratic_in_sequence_length():
    config = RunConfig(batch_size=1, precision=PrecisionMode.AMP)
    short = transformer_activation_bytes(_shape(), config, 1024)
    long = transformer_activation_bytes(_shape(), config, 2048)
    # Linear term doubles, quadratic term quadruples, so growth sits between 2x and 4x.
    assert 2.0 < long / short < 4.0


def test_gradient_checkpointing_reduces_activations():
    config = RunConfig(batch_size=4, seq_len=2048, precision=PrecisionMode.AMP)
    normal = transformer_activation_bytes(_shape(), config, 2048)
    checkpointed = transformer_activation_bytes(
        _shape(), config.replace(gradient_checkpointing=True), 2048
    )
    assert checkpointed < normal / 2


def test_missing_dimensions_widen_the_interval_instead_of_guessing():
    report = estimate(_profile(), RunConfig(batch_size=8))
    assert report.breakdown.activations == 0
    assert any("could not be estimated" in note for note in report.notes)

    # The interval must widen beyond what the profile's confidence alone would give,
    # so an unestimated term is visible rather than silently assumed to be zero.
    assert report.extra_uncertainty > 0
    low, high = report.interval
    confidence_only = 2 * report.profile.confidence.interval
    assert (high - low) / report.total > confidence_only


# --------------------------------------------------------------- risk bands


def test_bands_track_the_error_interval():
    gpu = hardware.GPUS["a100-80gb"]
    tiny = estimate(_profile(100_000_000), RunConfig(seq_len=128), gpu)
    huge = estimate(_profile(70_000_000_000), RunConfig(seq_len=128), gpu)
    assert tiny.band is RiskBand.FITS
    assert huge.band is RiskBand.CERTAIN_OOM


def test_unresolved_model_yields_unknown_band_not_a_number():
    profile = resolve_profile("not-a-real-model", allow_network=False)
    report = estimate(profile, RunConfig(), hardware.GPUS["a100-80gb"])
    assert report.band is RiskBand.UNKNOWN
    assert report.total == 0
    assert profile.confidence is Confidence.UNKNOWN


# ---------------------------------------------------------------- providers


def test_entry_point_failure_degrades_instead_of_raising():
    """Holds with or without the [vram] extra — only the wording of the reason differs.

    Asserting the exact message here made this test pass locally (torch installed) and
    fail in CI (torch absent), which is precisely the bug this module split exists to
    surface. The message-specific assertion lives in test_meta.py, where torch is
    guaranteed.
    """
    profile = resolve_profile("mypkg.models:build_gpt")
    assert profile.confidence is Confidence.UNKNOWN
    assert profile.reason


def test_offline_miss_suggests_online_lookup():
    profile = resolve_profile("some-org/some-model", allow_network=False)
    assert "--online" in profile.reason


# ----------------------------------------------------------------- extraction


TRAIN_SCRIPT = """
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loader = DataLoader(ds, batch_size=4, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32)

def train(device):
    for batch in loader:
        tokens = tokenizer(batch, max_length=2048)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**tokens).loss
        loss.backward()
        optimizer.step()
"""


def test_extraction_reads_the_whole_config():
    extracted = extract_from_source("train.py", textwrap.dedent(TRAIN_SCRIPT))
    config = extracted.config

    assert extracted.model_ref == "meta-llama/Llama-2-7b-hf"
    assert config.batch_size == 4          # the train loader, not the val loader
    assert config.seq_len == 2048
    assert config.precision is PrecisionMode.AMP
    assert config.optimizer is OptimizerKind.ADAMW
    assert config.inference_only is False


def test_extraction_records_provenance():
    extracted = extract_from_source("train.py", textwrap.dedent(TRAIN_SCRIPT))
    assert "train.py:" in extracted.config.sources["batch_size"]
    assert "train.py:" in extracted.config.sources["optimizer"]


def test_extraction_folds_module_constants():
    extracted = extract_from_source(
        "t.py",
        "BATCH = 64\nfrom torch.utils.data import DataLoader\n"
        "loader = DataLoader(ds, batch_size=BATCH)\n",
    )
    assert extracted.config.batch_size == 64


def test_extraction_detects_sharding_and_accumulation():
    extracted = extract_from_source(
        "t.py",
        "from torch.nn.parallel import DistributedDataParallel\n"
        "model = DistributedDataParallel(net)\n"
        "args = TrainingArguments(gradient_accumulation_steps=8, bf16=True)\n",
    )
    assert extracted.config.sharding is Sharding.DDP
    assert extracted.config.accumulation_steps == 8
    assert extracted.config.precision is PrecisionMode.AMP


def test_extraction_detects_checkpointing_and_flash():
    extracted = extract_from_source(
        "t.py",
        'model = AutoModel.from_pretrained("gpt2", attn_implementation="flash_attention_2")\n'
        "model.gradient_checkpointing_enable()\n",
    )
    assert extracted.config.flash_attention
    assert extracted.config.gradient_checkpointing


def test_inference_script_detected_by_absence_of_backward():
    extracted = extract_from_source(
        "t.py", 'model = AutoModel.from_pretrained("gpt2")\nout = model(x)\n'
    )
    assert extracted.config.inference_only


def test_unresolvable_model_argument_is_reported_not_guessed():
    from torch_preflight.vram import estimate_script

    report = estimate_script(
        "t.py",
        "model = AutoModel.from_pretrained(cfg.model_name)\nloss.backward()\n",
        gpu="a100-80gb",
    )
    assert report.band is RiskBand.UNKNOWN
    assert "computed at runtime" in report.profile.reason


# -------------------------------------------------------------------- solver


def test_solver_finds_a_fitting_configuration():
    gpu = hardware.GPUS["a100-80gb"]
    profile = archdb.resolve("llama-2-7b")
    config = RunConfig(batch_size=1, seq_len=512, precision=PrecisionMode.FP32)
    report = estimate(profile, config, gpu)
    assert report.band.is_failure

    options = solve(report)
    assert options and options[0].fits
    assert options[0].new_total < report.total


def test_solver_never_stacks_conflicting_changes():
    """You cannot switch the optimizer to both 8-bit AdamW and Adafactor."""
    gpu = hardware.GPUS["rtx3090"]
    profile = archdb.resolve("llama-2-7b")
    report = estimate(profile, RunConfig(batch_size=1, seq_len=1024), gpu)
    for option in solve(report):
        assert not ("8-bit AdamW" in option.label and "Adafactor" in option.label)
        assert not ("autocast" in option.label and "pure bf16" in option.label)


def test_solver_caps_the_stack_rather_than_listing_everything():
    gpu = hardware.GPUS["rtx3060"]
    profile = archdb.resolve("llama-2-70b")
    report = estimate(profile, RunConfig(batch_size=8, seq_len=4096), gpu)
    for option in solve(report):
        assert option.label.count(" + ") < MAX_STACK


def test_solver_is_silent_without_a_target_gpu():
    report = estimate(_profile(), RunConfig())
    assert solve(report) == []


# --------------------------------------------------------------------- TG010

_FINETUNE = """
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loader = DataLoader(ds, batch_size=4, num_workers=8, pin_memory=True)

def train(device):
    for batch in loader:
        tokens = tokenizer(batch, max_length=2048)
        optimizer.zero_grad(set_to_none=True)
        loss = model(**tokens).loss
        loss.backward()
        optimizer.step()
"""


def _tg010_codes(source, **cfg_kwargs):
    from torch_preflight.config import Config
    from torch_preflight.engine import check_source

    diagnostics, _ = check_source("t.py", textwrap.dedent(source), Config(**cfg_kwargs))
    return [d for d in diagnostics if d.code == "TG010"]


def test_tg010_silent_without_a_target_gpu():
    """No declared target means no opinion — the rule must not fire by default."""
    assert _tg010_codes(_FINETUNE) == []


def test_tg010_fires_when_the_run_will_not_fit():
    found = _tg010_codes(_FINETUNE, target_gpu="rtx4090")
    assert len(found) == 1
    assert "will OOM" in found[0].message
    assert "llama-2-7b" in found[0].message


def test_tg010_silent_when_the_run_fits():
    small = _FINETUNE.replace("meta-llama/Llama-2-7b-hf", "distilbert-base-uncased")
    assert _tg010_codes(small, target_gpu="a100-80gb") == []


def test_tg010_silent_when_the_model_cannot_be_identified():
    """An unresolvable model must never fail a build (RFC 0001 §8)."""
    source = _FINETUNE.replace('"meta-llama/Llama-2-7b-hf"', "cfg.model_name")
    assert _tg010_codes(source, target_gpu="rtx4090") == []


def test_tg010_silent_on_an_unknown_gpu():
    assert _tg010_codes(_FINETUNE, target_gpu="rtx9090") == []


def test_tg010_never_reaches_the_network(monkeypatch):
    """`check` must stay hermetic: no hub lookup, ever."""
    from torch_preflight.vram.providers import hub

    def explode(*args, **kwargs):
        raise AssertionError("check reached the network")

    monkeypatch.setattr(hub, "fetch_config", explode)
    source = _FINETUNE.replace("meta-llama/Llama-2-7b-hf", "some-org/unknown-model")
    assert _tg010_codes(source, target_gpu="rtx4090") == []
