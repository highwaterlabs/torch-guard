"""Calibration: does the cost model actually match reality?

Fixtures live in ``tests/calibration/``. See its README for what is measured, what is not,
and how to add a real measurement.
"""

import json
from pathlib import Path

import pytest

from torch_preflight.vram import archdb, hardware
from torch_preflight.vram.costmodel import estimate, params_from_transformer_shape
from torch_preflight.vram.types import (
    OptimizerKind,
    PrecisionMode,
    RunConfig,
    Sharding,
)

FIXTURES = Path(__file__).parent / "calibration"


def _load(name):
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        return json.load(handle)


# ------------------------------------------------------- parameter counts (enforced)

_PARAMS = _load("param_counts.json")


@pytest.mark.parametrize("model,published", sorted(_PARAMS["models"].items()))
def test_param_formula_matches_published_counts(model, published):
    profile = archdb.resolve(model)
    assert profile is not None, f"{model} missing from the bundled snapshot"
    assert profile.shape is not None, f"{model} has no architecture dimensions"

    computed = params_from_transformer_shape(profile.shape)
    error = abs(computed - published) / published
    assert error <= _PARAMS["tolerance"], (
        f"{model}: formula gives {computed:,}, published is {published:,} "
        f"({error:.1%} off, tolerance {_PARAMS['tolerance']:.0%})"
    )


def test_snapshot_counts_match_published_counts():
    """The snapshot's own numbers must agree with the fixture."""
    for model, published in _PARAMS["models"].items():
        profile = archdb.resolve(model)
        assert profile.param_count == published, model


# ------------------------------------------------- definitional accounting (enforced)


def test_mixed_precision_adamw_accounting():
    """The ZeRO-paper accounting for fp16 params + fp32 master + Adam states.

    2 (weights) + 2 (grads) + 4 (master) + 8 (optimizer) = 16 bytes per parameter.
    This is arithmetic, not measurement, so it is exact.
    """
    params = 1_000_000_000
    profile = archdb.resolve("llama-2-7b")
    profile.param_count = profile.trainable_params = params

    report = estimate(
        profile,
        RunConfig(precision=PrecisionMode.FP16_MASTER, optimizer=OptimizerKind.ADAMW),
    )
    per_param = (
        report.breakdown.weights
        + report.breakdown.gradients
        + report.breakdown.master_weights
        + report.breakdown.optimizer_state
    ) / params
    assert per_param == pytest.approx(16.0)


def test_zero3_divides_state_by_world_size():
    profile = archdb.resolve("llama-2-7b")
    single = estimate(profile, RunConfig(seq_len=512))
    sharded = estimate(
        profile, RunConfig(seq_len=512, sharding=Sharding.ZERO3, world_size=8)
    )
    for term in ("weights", "gradients", "optimizer_state"):
        assert getattr(sharded.breakdown, term) == pytest.approx(
            getattr(single.breakdown, term) / 8, rel=0.01
        )


# ------------------------------------------------------ measured peaks (aspirational)

_MEASURED = _load("measured_peaks.json")


@pytest.mark.skipif(
    not _MEASURED["runs"],
    reason="no measured peaks recorded yet — see tests/calibration/README.md",
)
@pytest.mark.parametrize("run", _MEASURED["runs"], ids=lambda r: r.get("model", "?"))
def test_estimate_matches_measured_peak(run):
    profile = archdb.resolve(run["model"])
    assert profile is not None, f"{run['model']} not in the snapshot"

    raw = dict(run["config"])
    config = RunConfig(
        batch_size=raw.get("batch_size", 1),
        seq_len=raw.get("seq_len"),
        image_size=raw.get("image_size"),
        precision=PrecisionMode(raw.get("precision", "fp32")),
        optimizer=OptimizerKind(raw.get("optimizer", "adamw")),
        gradient_checkpointing=raw.get("gradient_checkpointing", False),
        flash_attention=raw.get("flash_attention", False),
        world_size=raw.get("world_size", 1),
        sharding=Sharding(raw.get("sharding", "none")),
    )

    gpu, _ = hardware.resolve(run["gpu"])
    report = estimate(profile, config, gpu)

    measured = run["measured_peak_bytes"]
    error = abs(report.total - measured) / measured
    assert error <= _MEASURED["tolerance"], (
        f"{run['model']}: estimated {report.total:,} vs measured {measured:,} "
        f"({error:.1%} off, tolerance {_MEASURED['tolerance']:.0%})"
    )


def test_measured_fixture_file_is_wellformed():
    """Guards the fixture format even while the run list is empty."""
    assert isinstance(_MEASURED["runs"], list)
    assert 0 < _MEASURED["tolerance"] < 1
    for run in _MEASURED["runs"]:
        assert {"model", "gpu", "config", "measured_peak_bytes"} <= set(run)


# ---------------------------------------------- activation coefficients (measured)

_ACTIVATIONS_FILE = FIXTURES / "measured_activations.json"


@pytest.mark.skipif(
    not _ACTIVATIONS_FILE.exists(),
    reason="run tests/calibration/measure_activations.py to generate",
)
def test_activation_constants_match_the_measurement():
    """The constants in costmodel.py must be the ones we actually measured.

    Changing a constant without re-running the measurement should fail here — that is the
    whole point of the fixture (see calibration/README.md).
    """
    from torch_preflight.vram import costmodel

    measured = _load("measured_activations.json")["results"]

    assert costmodel.ACT_LINEAR_COEFF_NO_DROPOUT == pytest.approx(
        measured["minimal"]["act_linear_coeff"], abs=0.5
    )
    assert costmodel.ACT_ATTN_COEFF_NO_DROPOUT == pytest.approx(
        measured["minimal"]["act_attn_coeff"], abs=0.5
    )
    assert costmodel.ACT_LINEAR_COEFF_DROPOUT == pytest.approx(
        measured["realistic"]["act_linear_coeff"], abs=0.5
    )
    assert costmodel.ACT_ATTN_COEFF_DROPOUT == pytest.approx(
        measured["realistic"]["act_attn_coeff"], abs=0.5
    )
    assert costmodel.CHECKPOINT_ACT_COEFF == pytest.approx(
        measured["checkpointing"]["checkpoint_act_coeff"], abs=0.5
    )


@pytest.mark.skipif(
    not _ACTIVATIONS_FILE.exists(),
    reason="run tests/calibration/measure_activations.py to generate",
)
def test_flash_measurement_shows_no_residual_quadratic_term():
    measured = _load("measured_activations.json")["results"]
    largest = max(measured["flash"]["measured_bytes_by_seq"].values())
    # The fitted s^2 coefficient should be negligible next to the total.
    assert abs(measured["flash"]["residual_quadratic"]) < largest * 1e-4


@pytest.mark.skipif(
    not _ACTIVATIONS_FILE.exists(),
    reason="run tests/calibration/measure_activations.py to generate",
)
def test_measurement_fit_had_no_unexplained_constant():
    """A non-zero constant term means something sequence-independent leaked into the fit.

    That is exactly the bug that made the first measurement overstate the linear
    coefficient by ~2x: linear layers save their weights, which are not activations.
    """
    measured = _load("measured_activations.json")["results"]
    for variant in ("minimal", "realistic"):
        largest = max(measured[variant]["measured_bytes_by_seq"].values())
        assert abs(measured[variant]["fit_constant_bytes"]) < largest * 1e-3


# ----------------------------------------- encoder-decoder coefficients (measured)

_ENCDEC_FILE = FIXTURES / "measured_encoder_decoder.json"


@pytest.mark.skipif(
    not _ENCDEC_FILE.exists(),
    reason="run tests/calibration/measure_encoder_decoder.py to generate",
)
def test_encoder_decoder_constants_match_the_measurement():
    """Changing a shipped coefficient without re-measuring must fail here."""
    from torch_preflight.vram import costmodel

    measured = _load("measured_encoder_decoder.json")["families"]
    for family, shipped in costmodel.ENCODER_DECODER_COEFFS.items():
        assert family in measured, f"{family} has no measurement behind it"
        for key in ("enc_linear", "dec_linear", "cross_kv"):
            assert shipped[key] == pytest.approx(measured[family][key], abs=0.5), (
                f"{family}.{key} is {shipped[key]} but the fixture measured "
                f"{measured[family][key]}"
            )


@pytest.mark.skipif(
    not _ENCDEC_FILE.exists(),
    reason="run tests/calibration/measure_encoder_decoder.py to generate",
)
def test_encoder_decoder_fit_residuals_stayed_small():
    """A large residual means the functional form is wrong, not just the constants."""
    for family, result in _load("measured_encoder_decoder.json")["families"].items():
        assert result["encoder_residual_max_pct"] < 10, family
        assert result["decoder_residual_max_pct"] < 10, family


@pytest.mark.skipif(
    not _ENCDEC_FILE.exists(),
    reason="run tests/calibration/measure_encoder_decoder.py to generate",
)
def test_decoder_layers_cost_more_than_encoder_layers():
    """A physical sanity check the degenerate fits all failed.

    A decoder layer carries self-attention *and* cross-attention, so it must retain more
    per layer than an encoder layer. The unconstrained fit returned a Whisper decoder
    coefficient of 0.16 -- lower than its encoder -- which is what exposed the
    collinearity rather than any residual.
    """
    for family, result in _load("measured_encoder_decoder.json")["families"].items():
        assert result["dec_linear"] > result["enc_linear"], family


# ------------------------------------------- the CUDA harness itself (no GPU needed)

torch = pytest.importorskip("torch", reason="calibration harness needs torch")


#: The measurement harness ships alongside the fixtures it produces, so contributors can
#: regenerate them from a plain clone.
CUDA_HARNESS = FIXTURES / "measure_cuda.py"


def _load_cuda_harness():
    import importlib.util

    spec = importlib.util.spec_from_file_location("measure_cuda", CUDA_HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_harness_imports_without_a_gpu():
    """It must be importable anywhere; the GPU check belongs in main(), not at import."""
    module = _load_cuda_harness()
    assert hasattr(module, "measure_cuda_context")
    assert hasattr(module, "NO_CUDA_MESSAGE")


def test_cuda_harness_exits_cleanly_without_a_gpu(capsys):
    module = _load_cuda_harness()
    if torch.cuda.is_available():  # pragma: no cover - not our CI
        pytest.skip("this machine has CUDA")
    assert module.main([]) == 1
    assert "No CUDA device" in capsys.readouterr().err


@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("flash", [False, True])
def test_cuda_harness_model_trains_on_cpu(dropout, flash):
    """Catch a broken forward/backward here rather than after pasting into Colab."""
    module = _load_cuda_harness()
    model = module.Stack(2, 64, 4, 256, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.randn(2, 16, 64), flash=flash).pow(2).mean()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_cuda_harness_model_matches_the_analytic_param_formula():
    """The harness and the cost model must describe the same architecture."""
    from torch_preflight.vram.types import TransformerShape

    module = _load_cuda_harness()
    model = module.Stack(4, 256, 8, 1024, 0.0)
    actual = sum(p.numel() for p in model.parameters())
    formula = params_from_transformer_shape(
        TransformerShape(
            layers=4, hidden=256, heads=8, intermediate=1024,
            vocab=0, tied_embeddings=True,
        )
    )
    assert formula == actual


@pytest.mark.parametrize(
    "name,total_gib,expected",
    [
        ("Tesla T4", 15, "t4"),
        ("NVIDIA A100-SXM4-40GB", 40, "a100-40gb"),
        ("NVIDIA A100-SXM4-80GB", 80, "a100-80gb"),
        ("Tesla V100-SXM2-32GB", 32, "v100-32gb"),
        ("NVIDIA GeForce RTX 4090", 24, "rtx4090"),
        ("Some Unreleased Card", 48, "unknown"),
    ],
)
def test_cuda_harness_maps_device_names(name, total_gib, expected):
    """Capacity disambiguates parts that ship in two sizes."""
    module = _load_cuda_harness()
    assert module._gpu_key(name, total_gib * 1024 ** 3) == expected


# --------------------------------------------- CUDA-allocator constants (measured)

_CUDA_FILE = FIXTURES / "measured_cuda.json"


@pytest.mark.skipif(not _CUDA_FILE.exists(), reason="no CUDA measurement recorded")
def test_cuda_constants_match_the_measurement():
    from torch_preflight.vram import costmodel

    adopted = _load("measured_cuda.json")["adopted"]
    assert costmodel.CUDA_CONTEXT_BYTES == adopted["cuda_context_mib"] * 1024 ** 2
    assert costmodel.FRAGMENTATION_FRACTION == pytest.approx(
        adopted["fragmentation_fraction"], abs=0.005
    )


@pytest.mark.skipif(not _CUDA_FILE.exists(), reason="no CUDA measurement recorded")
def test_measured_gpu_carries_its_own_context():
    """Where we have a per-card number, that card should use it."""
    from torch_preflight.vram import hardware

    measurement = _load("measured_cuda.json")["measurements"][0]
    gpu = hardware.GPUS[measurement["gpu_key"]]
    assert gpu.context_mib is not None
    assert gpu.context_bytes == gpu.context_mib * 1024 ** 2


@pytest.mark.skipif(not _MEASURED["runs"], reason="no measured peaks recorded")
def test_mean_error_against_measured_peaks_stays_small():
    """Per-run tolerance is loose enough for the worst case; this guards the typical one.

    A regression that degraded most estimates while staying inside the per-run band would
    slip past ``test_estimate_matches_measured_peak``. This catches it.
    """
    errors = []
    for run in _MEASURED["runs"]:
        profile = archdb.resolve(run["model"])
        raw = run["config"]
        config = RunConfig(
            batch_size=raw["batch_size"],
            seq_len=raw.get("seq_len"),
            image_size=raw.get("image_size"),
            precision=PrecisionMode(raw["precision"]),
            optimizer=OptimizerKind(raw["optimizer"]),
            gradient_checkpointing=raw["gradient_checkpointing"],
            flash_attention=raw["flash_attention"],
        )
        gpu, _ = hardware.resolve(run["gpu"])
        report = estimate(profile, config, gpu)
        errors.append(abs(report.total - run["measured_peak_bytes"]) / run["measured_peak_bytes"])

    mean_error = sum(errors) / len(errors)
    assert mean_error < 0.10, f"mean absolute error {mean_error:.1%} across {len(errors)} runs"


# ------------------------------------------------- CNN activations (measured on meta)

_CNN_FILE = FIXTURES / "measured_cnn_activations.json"


@pytest.mark.skipif(not _CNN_FILE.exists(), reason="run measure_cnn_activations.py")
def test_cnn_activations_in_the_snapshot_match_the_measurement():
    """Vision models used to report activations as unknown. They are now measured."""
    measured = _load("measured_cnn_activations.json")
    for name, m in measured["models"].items():
        profile = archdb.resolve(name)
        assert profile is not None, f"{name} missing from the snapshot"
        assert profile.param_count == m["params"], name
        assert profile.activation_bytes_per_sample == m["activation_bytes_per_sample"], name


@pytest.mark.skipif(not _CNN_FILE.exists(), reason="run measure_cnn_activations.py")
def test_cnn_scaling_assumptions_held_when_measured():
    """The cost model scales a per-sample figure by batch and by spatial area.

    Both were verified at measurement time rather than assumed; if either stopped holding
    the stored number would be meaningless.
    """
    for name, m in _load("measured_cnn_activations.json")["models"].items():
        assert m["batch_linear"], f"{name}: activations are not linear in batch size"
        assert m["area_scaled"], f"{name}: activations do not scale with spatial area"


def test_vision_models_now_produce_an_activation_estimate():
    """The regression this closes: a ResNet used to widen the interval and say nothing."""
    from torch_preflight.vram.costmodel import estimate

    report = estimate(
        archdb.resolve("resnet50"),
        RunConfig(batch_size=64, image_size=224, precision=PrecisionMode.AMP),
        hardware.GPUS["rtx4090"],
    )
    assert report.breakdown.activations > 0
    assert not any("could not be estimated" in n for n in report.notes)


def test_activation_memory_is_not_predictable_from_parameter_count():
    """Why these had to be measured rather than derived.

    MobileNet-V2 has 3.5M parameters and more activation memory than VGG-16, which has
    138M. Any formula keyed on parameter count would be badly wrong for both.
    """
    mobilenet = archdb.resolve("mobilenet_v2")
    vgg = archdb.resolve("vgg16")
    assert mobilenet.param_count < vgg.param_count / 30
    assert mobilenet.activation_bytes_per_sample > vgg.activation_bytes_per_sample


# ------------------------------------------------------ LM head (measured vs fitted)


def test_lm_head_retained_bytes_match_the_meta_measurement():
    """4 bytes per logit element, measured across five shapes and three vocabularies.

    Precision-independent: cross_entropy upcasts to fp32 whatever autocast is doing.
    """
    from torch_preflight.vram import costmodel

    assert costmodel.LM_HEAD_RETAINED_BYTES == 4


def test_lm_head_cost_does_not_vary_with_precision():
    """The retained copy is fp32 regardless, so AMP must not shrink this term."""
    from torch_preflight.vram.costmodel import lm_head_bytes
    from torch_preflight.vram.types import TransformerShape

    shape = TransformerShape(layers=2, hidden=256, heads=4, vocab=50257, has_lm_head=True)
    fp32 = lm_head_bytes(shape, RunConfig(batch_size=2, precision=PrecisionMode.FP32), 128)
    amp = lm_head_bytes(shape, RunConfig(batch_size=2, precision=PrecisionMode.AMP), 128)
    assert fp32 == amp


def test_models_without_an_lm_head_are_not_charged_for_one():
    from torch_preflight.vram.costmodel import lm_head_bytes
    from torch_preflight.vram.types import TransformerShape

    encoder = TransformerShape(layers=2, hidden=256, heads=4, vocab=30522)
    assert lm_head_bytes(encoder, RunConfig(batch_size=2), 128) == 0
