"""VRAM estimation: hardware DB, cost model, extraction, providers, solver."""

import json
import pathlib
import subprocess
import sys
import textwrap
import warnings

import pytest

from torch_guard.vram import archdb, hardware
from torch_guard.vram.costmodel import (
    estimate,
    params_from_transformer_shape,
    transformer_activation_bytes,
)
from torch_guard.vram.extract import extract_from_source
from torch_guard.vram.providers import resolve_profile, static
from torch_guard.vram.solver import MAX_STACK, solve
from torch_guard.vram.types import (
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

    One careless top-level ``import torch`` would silently make `pip install torch-guard`
    pull in gigabytes. Normal tests cannot catch it, so this asserts it directly.
    """
    script = (
        "import sys, torch_guard;"
        "from torch_guard import vram;"
        "torch_guard.check_source('t.py', 'x = 1');"
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


def test_unimportable_entry_point_is_reported_not_raised():
    profile = resolve_profile("mypkg.models:build_gpt")
    assert profile.confidence is Confidence.UNKNOWN
    assert "could not import" in profile.reason


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
    from torch_guard.vram import estimate_script

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
    from torch_guard.config import Config
    from torch_guard.engine import check_source

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
    from torch_guard.vram.providers import hub

    def explode(*args, **kwargs):
        raise AssertionError("check reached the network")

    monkeypatch.setattr(hub, "fetch_config", explode)
    source = _FINETUNE.replace("meta-llama/Llama-2-7b-hf", "some-org/unknown-model")
    assert _tg010_codes(source, target_gpu="rtx4090") == []


# ------------------------------------------------- meta-device provider (phase 2)

torch = pytest.importorskip("torch", reason="meta provider needs the [vram] extra")

from torch_guard.vram.providers import meta  # noqa: E402
from meta_fixture_model import Tiny  # noqa: E402

FIXTURE = "meta_fixture_model"


def test_meta_provider_counts_parameters_exactly():
    """Checked against arithmetic, not against another copy of the same code."""
    profile = meta.profile(f"{FIXTURE}:Tiny")
    assert profile.param_count == Tiny.expected_params()
    assert profile.confidence is Confidence.HIGH
    assert profile.source == "meta-device"


def test_meta_provider_allocates_nothing():
    """No backing allocation, but storages still report their logical size.

    That combination is exactly what makes this technique work: ``data_ptr() == 0``
    means nothing was allocated, while ``nbytes()`` still reports what the tensor *would*
    occupy — which is the number we want. (It is also why dedup must key on
    ``storage._cdata``: every meta tensor shares a null data pointer.)
    """
    model = meta.build_model(f"{FIXTURE}:Tiny")
    assert all(p.device.type == "meta" for p in model.parameters())
    assert all(p.untyped_storage().data_ptr() == 0 for p in model.parameters())
    assert all(p.untyped_storage().nbytes() > 0 for p in model.parameters())


def test_meta_provider_honours_constructor_arguments():
    profile = meta.profile(f"{FIXTURE}:build_tiny", model_args={"vocab": 500, "hidden": 64})
    assert profile.param_count == Tiny.expected_params(vocab=500, hidden=64)


def test_meta_provider_separates_trainable_parameters():
    profile = meta.profile(f"{FIXTURE}:build_frozen")
    assert profile.param_count == Tiny.expected_params()
    assert profile.trainable_params < profile.param_count


def test_meta_provider_counts_buffers():
    profile = meta.profile(f"{FIXTURE}:Tiny")
    assert profile.buffer_bytes == 32 * 4  # one fp32 buffer of size `hidden`


def test_meta_provider_measures_activations_when_shapes_are_known():
    config = RunConfig(batch_size=4, seq_len=16)
    profile = meta.profile(f"{FIXTURE}:Tiny", config)
    assert profile.activation_bytes_per_sample is not None
    assert profile.activation_bytes_per_sample > 0


def test_meta_activations_scale_with_sequence_length():
    short = meta.profile(f"{FIXTURE}:Tiny", RunConfig(batch_size=2, seq_len=16))
    long = meta.profile(f"{FIXTURE}:Tiny", RunConfig(batch_size=2, seq_len=64))
    assert long.activation_bytes_per_sample == pytest.approx(
        short.activation_bytes_per_sample * 4, rel=0.05
    )


def test_meta_activations_are_none_without_an_input_shape():
    """No seq_len and no image_size means we cannot build an input, so we say so."""
    profile = meta.profile(f"{FIXTURE}:Tiny", RunConfig(batch_size=4))
    assert profile.activation_bytes_per_sample is None
    assert profile.param_count > 0  # parameters are still exact


def test_meta_activations_exclude_parameter_storages():
    """The trap from spike 0001: weights saved for backward are not activations."""
    model = meta.build_model(f"{FIXTURE}:Tiny")
    measured = meta.measure_activations(model, RunConfig(batch_size=1, seq_len=8))
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    assert measured is not None
    assert measured < parameter_bytes, "parameter storages leaked into the activation total"


@pytest.mark.parametrize(
    "reference,fragment",
    [
        ("meta_fixture_model", "not a valid entry point"),
        ("meta_fixture_model:nope", "has no attribute"),
        ("meta_fixture_model:NOT_CALLABLE", "not callable"),
        ("meta_fixture_model:not_a_model", "expected an nn.Module"),
        ("meta_fixture_model:build_tiny", "--model-args"),
    ],
)
def test_meta_provider_reports_bad_entry_points_clearly(reference, fragment):
    kwargs = {"bogus_argument": 1} if fragment == "--model-args" else None
    with pytest.raises(meta.EntryPointError, match=fragment):
        meta.build_model(reference, kwargs)


def test_resolve_profile_routes_entry_points_to_the_meta_provider():
    profile = resolve_profile(f"{FIXTURE}:Tiny", config=RunConfig(batch_size=2, seq_len=8))
    assert profile.source == "meta-device"
    assert profile.param_count == Tiny.expected_params()


def test_resolve_profile_turns_entry_point_failures_into_unknown():
    """A broken entry point must degrade, never crash the run."""
    profile = resolve_profile("meta_fixture_model:nope")
    assert profile.confidence is Confidence.UNKNOWN
    assert "has no attribute" in profile.reason


@pytest.mark.parametrize(
    "pairs,expected",
    [
        (["n=10"], {"n": 10}),
        (["width=1.5"], {"width": 1.5}),
        (["pretrained=false"], {"pretrained": False}),
        (["name=resnet50"], {"name": "resnet50"}),
        (["x=none"], {"x": None}),
    ],
)
def test_model_args_are_coerced(pairs, expected):
    assert meta.parse_model_args(pairs) == expected


def test_model_args_require_key_value():
    with pytest.raises(meta.EntryPointError, match="key=value"):
        meta.parse_model_args(["justakey"])


def test_meta_profile_feeds_the_cost_model():
    """End to end: an entry point produces a usable estimate."""
    from torch_guard.vram import estimate_config

    config = RunConfig(batch_size=8, seq_len=128, precision=PrecisionMode.AMP)
    profile = meta.profile(f"{FIXTURE}:Tiny", config)
    report = estimate_config(profile, config, gpu="a100-80gb")

    assert report.band is RiskBand.FITS
    assert report.breakdown.weights > 0
    assert report.breakdown.activations > 0


def test_meta_measurement_agrees_with_the_analytic_formula():
    """Two independent methods on the same architecture must produce the same count.

    The formula is validated against published parameter counts in the calibration
    suite; the meta provider measures the real module. Agreement means the "exact"
    claim holds and the snapshot's dimensions describe what they say they do.
    """
    profile = meta.profile(f"{FIXTURE}:MiniTransformer")
    derived = params_from_transformer_shape(
        TransformerShape(
            layers=3, hidden=64, heads=4, intermediate=256, vocab=200,
            tied_embeddings=False,
        )
    )
    # The formula includes layer-norm terms the fixture also has, so this is exact
    # up to the bias-free variations it models generically.
    error = abs(profile.param_count - derived) / profile.param_count
    assert error < 0.02, f"meta measured {profile.param_count:,}, formula gives {derived:,}"


# ------------------------------------------------- autodetection (RFC 0001 layer 2)

from torch_guard.analysis.context import build_context  # noqa: E402
from torch_guard.vram import autodetect as autodetect_mod  # noqa: E402

assert hasattr(autodetect_mod, 'module_is_import_safe'), 'submodule shadowed by a function'


def _detect(source):
    return autodetect_mod.autodetect(build_context("train.py", textwrap.dedent(source)))


SAFE_SCRIPT = """
import torch
from models import Classifier

NUM_CLASSES = 100

def main():
    model = Classifier(num_classes=NUM_CLASSES, hidden=2048)
    loss.backward()

if __name__ == "__main__":
    main()
"""


def test_autodetect_resolves_class_and_folds_constants():
    detected = _detect(SAFE_SCRIPT)
    assert detected.ok
    assert detected.reference == "models:Classifier"
    assert detected.kwargs == {"num_classes": 100, "hidden": 2048}


def test_autodetect_names_the_argument_that_blocked_it():
    """Layer 3: report what stopped us, never guess a value."""
    detected = _detect(
        """
        from models import Classifier
        def main():
            model = Classifier(num_classes=cfg.num_classes)
        """
    )
    assert not detected.ok
    assert "cfg.num_classes" in detected.reason
    assert "--model" in detected.reason


def test_autodetect_reports_when_nothing_looks_like_a_model():
    detected = _detect("x = 1\ny = compute(x)\n")
    assert not detected.ok
    assert "No model construction found" in detected.reason


@pytest.mark.parametrize(
    "top_level,safe",
    [
        ("CONST = 5", True),
        ("import os", True),
        ('"""a docstring"""', True),
        ('if __name__ == "__main__":\n    main()', True),
        ("dataset = load_dataset('huge')", False),
        ("model = Classifier()", False),
        ("connect()", False),
    ],
)
def test_import_safety_gate(top_level, safe):
    """Importing a module runs its top level, so anything that does work is refused."""
    import libcst as cst

    # Built without dedent: a multi-line snippet would break the common-prefix maths.
    source = "import torch.nn as nn\n\nclass Classifier(nn.Module):\n    pass\n\n" + top_level + "\n"
    assert autodetect_mod.module_is_import_safe(cst.parse_module(source)) is safe


def test_autodetect_refuses_a_script_that_works_on_import():
    """The class is local, but importing the file would build the model for real."""
    detected = _detect(
        """
        import torch.nn as nn

        class Classifier(nn.Module):
            def __init__(self, n=10):
                super().__init__()
                self.fc = nn.Linear(4, n)

        model = Classifier(n=10)
        """
    )
    assert not detected.ok
    assert "does work on import" in detected.reason


def test_autodetect_accepts_a_local_class_in_an_inert_script():
    detected = _detect(
        """
        import torch.nn as nn

        class Classifier(nn.Module):
            def __init__(self, n=10):
                super().__init__()
                self.fc = nn.Linear(4, n)

        def main():
            model = Classifier(n=32)

        if __name__ == "__main__":
            main()
        """
    )
    assert detected.ok
    assert detected.reference.endswith(":Classifier")
    assert detected.kwargs == {"n": 32}


def test_autodetect_prefers_a_real_module_subclass_over_a_name_guess():
    detected = _detect(
        """
        import torch.nn as nn
        from thirdparty import Whatever

        class Encoder(nn.Module):
            pass

        def main():
            model = Whatever(1)
            encoder = Encoder()
        """
    )
    assert detected.construction.class_name == "Encoder"


@pytest.mark.parametrize(
    "literal,expected",
    [("1", 1), ("1.5", 1.5), ("True", True), ("None", None), ("-3", -3), ("[1, 2]", [1, 2])],
)
def test_literal_folding(literal, expected):
    import libcst as cst

    node = cst.parse_expression(literal)
    assert autodetect_mod.fold_literal(node, {}) == expected


def test_literal_folding_refuses_computed_values():
    import libcst as cst

    with pytest.raises(ValueError):
        autodetect_mod.fold_literal(cst.parse_expression("get_size()"), {})


def test_estimate_script_autodetects_end_to_end(tmp_path, monkeypatch):
    from torch_guard.vram import estimate_script

    (tmp_path / "mymodels.py").write_text(
        textwrap.dedent(
            """
            import torch.nn as nn

            class Net(nn.Module):
                def __init__(self, hidden=64):
                    super().__init__()
                    self.fc = nn.Linear(hidden, hidden)
            """
        )
    )
    script = tmp_path / "train.py"
    script.write_text(
        textwrap.dedent(
            """
            from mymodels import Net

            def main():
                model = Net(hidden=128)
                loss.backward()
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    report = estimate_script(str(script), gpu="a100-80gb")

    assert report.profile.source == "meta-device"
    assert report.profile.param_count == 128 * 128 + 128


# ------------------------------------------------------- hub config mapping (offline)

from torch_guard.vram.providers import hub as hub_provider  # noqa: E402


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


# --------------------------------------------------------- VRAMGuard (phase 3)

from torch_guard.vram.guard import VRAMGuard, VramRiskError  # noqa: E402


def _guard_model(hidden=256):
    return torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
    )


def test_vram_guard_is_lazily_exported():
    """Available as `from torch_guard import VRAMGuard` without torch at import time."""
    import torch_guard

    assert torch_guard.VRAMGuard is VRAMGuard
    assert "VRAMGuard" in dir(torch_guard)


def test_guard_counts_live_parameters_exactly():
    from torch_guard.vram.guard import profile_live_model

    model = _guard_model(hidden=64)
    profile = profile_live_model(model)
    assert profile.param_count == sum(p.numel() for p in model.parameters())
    assert profile.confidence is Confidence.HIGH


def test_guard_raises_when_the_run_cannot_fit():
    model = _guard_model(hidden=1024)
    with pytest.raises(VramRiskError) as excinfo:
        with VRAMGuard(model, max_vram="4MiB", batch_size=8):
            pass
    assert "projected to need" in str(excinfo.value)
    assert excinfo.value.report.band is RiskBand.CERTAIN_OOM


def test_guard_is_silent_when_the_run_fits():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        with VRAMGuard(_guard_model(hidden=64), max_vram="8GiB", batch_size=1) as guard:
            pass
    assert guard.report.band is RiskBand.FITS


def _limit_near_projection(model, ratio):
    """A max_vram sized relative to what this model is actually projected to need.

    Derived rather than hard-coded, so the test still targets the intended band when a
    calibration constant changes.
    """
    probe = VRAMGuard(model, max_vram="1TiB", batch_size=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with probe:
            pass
    # custom_gpu keeps a 2% driver reserve, so scale the limit to hit the usable target.
    return str(int(probe.report.total * ratio / 0.98))


def test_guard_warns_rather_than_raising_when_merely_likely():
    """Aborting a job on a guess is worse than the OOM it would prevent."""
    model = _guard_model(hidden=1024)
    guard = VRAMGuard(model, max_vram=_limit_near_projection(model, 0.97), batch_size=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with guard:
            pass
    assert guard.report.band is RiskBand.LIKELY_OOM
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_guard_warns_on_a_tight_fit():
    model = _guard_model(hidden=1024)
    guard = VRAMGuard(model, max_vram=_limit_near_projection(model, 1.05), batch_size=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with guard:
            pass
    assert guard.report.band is RiskBand.TIGHT
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_guard_strict_mode_raises_on_likely_failure():
    model = _guard_model(hidden=1024)
    limit = _limit_near_projection(model, 0.97)
    with pytest.raises(VramRiskError):
        with VRAMGuard(model, max_vram=limit, batch_size=1, strict=True):
            pass


def test_guard_reports_when_activations_were_not_estimated():
    model = _guard_model(hidden=1024)
    with pytest.raises(VramRiskError) as excinfo:
        with VRAMGuard(model, max_vram="1MiB", batch_size=4):
            pass
    assert "activations were not estimated" in str(excinfo.value)


def test_guard_does_nothing_without_a_device_or_limit():
    """No CUDA and no max_vram means nothing to compare against — stay quiet."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with VRAMGuard(_guard_model(), batch_size=4) as guard:
            pass
    assert guard.report is not None
    assert guard.report.band is RiskBand.UNKNOWN


@pytest.mark.parametrize(
    "factory,expected",
    [
        (lambda p: torch.optim.SGD(p, lr=0.1), OptimizerKind.SGD),
        (lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9), OptimizerKind.SGD_MOMENTUM),
        (lambda p: torch.optim.AdamW(p, lr=1e-3), OptimizerKind.ADAMW),
        (lambda p: torch.optim.Adam(p, lr=1e-3), OptimizerKind.ADAM),
        (lambda p: torch.optim.RMSprop(p, lr=1e-3), OptimizerKind.RMSPROP),
    ],
)
def test_guard_infers_the_optimizer(factory, expected):
    model = _guard_model(hidden=32)
    guard = VRAMGuard(model, optimizer=factory(model.parameters()), max_vram="8GiB")
    assert guard.config.optimizer is expected


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (torch.float32, PrecisionMode.FP32),
        (torch.bfloat16, PrecisionMode.PURE_BF16),
        (torch.float16, PrecisionMode.PURE_FP16),
    ],
)
def test_guard_reads_precision_off_the_model(dtype, expected):
    model = _guard_model(hidden=32).to(dtype)
    assert VRAMGuard(model, max_vram="8GiB").config.precision is expected


def test_guard_explicit_precision_wins():
    model = _guard_model(hidden=32)
    guard = VRAMGuard(model, max_vram="8GiB", precision="amp")
    assert guard.config.precision is PrecisionMode.AMP


def test_guard_rejects_an_unparseable_limit():
    with pytest.raises(ValueError, match="max_vram"):
        with VRAMGuard(_guard_model(), max_vram="loads"):
            pass


def test_guard_accuracy_is_none_without_cuda():
    with VRAMGuard(_guard_model(), max_vram="8GiB") as guard:
        pass
    if not torch.cuda.is_available():
        assert guard.measured_peak is None
        assert guard.accuracy is None


def test_guard_suggests_a_remediation_when_it_can():
    model = _guard_model(hidden=2048)
    with pytest.raises(VramRiskError) as excinfo:
        with VRAMGuard(model, max_vram="8MiB", batch_size=8,
                       optimizer=torch.optim.AdamW(model.parameters())):
            pass
    assert excinfo.value.report.remediations


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2TiB", 2 * 1024 ** 4),
        ("1tb", 1024 ** 4),
        ("48GiB", 48 * GIB),
        ("512MiB", 512 * 1024 ** 2),
        ("178257920", 178257920),
        ("nope", None),
    ],
)
def test_parse_memory_units(text, expected):
    assert hardware.parse_memory(text) == expected


# ------------------------------------------- hub integration (real configs, offline)

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
    from torch_guard.vram import estimate_config

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
