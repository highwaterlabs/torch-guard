"""VRAMGuard — the runtime check. Requires the ``[vram]`` extra."""

import warnings

import pytest

torch = pytest.importorskip("torch", reason="VRAMGuard needs the [vram] extra")

from torch_preflight.vram.guard import VRAMGuard, VramRiskError  # noqa: E402
from torch_preflight.vram import hardware  # noqa: E402
from torch_preflight.vram.types import (  # noqa: E402
    GIB,
    Confidence,
    OptimizerKind,
    PrecisionMode,
    RiskBand,
)



def _guard_model(hidden=256):
    return torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
    )


def test_vram_guard_is_lazily_exported():
    """Available as `from torch_preflight import VRAMGuard` without torch at import time."""
    import torch_preflight

    assert torch_preflight.VRAMGuard is VRAMGuard
    assert "VRAMGuard" in dir(torch_preflight)


def test_guard_counts_live_parameters_exactly():
    from torch_preflight.vram.guard import profile_live_model

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




# --------------------------------------------------------------- measured activations


def _conv_model():
    """A small conv net: activations dominate, which is the case that used to be missed."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
    )


def test_guard_measures_activations_when_given_an_image_size():
    """The term the guard used to leave at zero is now measured from the live module."""
    guard = VRAMGuard(_conv_model(), max_vram="8GiB", batch_size=8, image_size=32,
                      verify=False)
    assert guard.activation_source == "measured"
    assert guard.profile.activation_bytes_per_sample > 0
    with guard:
        pass
    assert guard.report.breakdown.activations > 0


def test_measured_activations_scale_with_the_declared_batch_size():
    """Stored per-sample, so doubling the batch doubles the activation term exactly."""
    small = VRAMGuard(_conv_model(), max_vram="8GiB", batch_size=4, image_size=32,
                      verify=False)
    large = VRAMGuard(_conv_model(), max_vram="8GiB", batch_size=8, image_size=32,
                      verify=False)
    assert small.profile.activation_bytes_per_sample == large.profile.activation_bytes_per_sample
    with small, large:
        pass
    assert large.report.breakdown.activations == 2 * small.report.breakdown.activations


def test_measuring_activations_does_not_touch_the_model():
    """It runs on meta parameters via functional_call — the caller's model is untouched."""
    model = _conv_model()
    model.eval()
    before = [(p.device, p.dtype, p.requires_grad, float(p.detach().sum())) for p in model.parameters()]
    VRAMGuard(model, max_vram="8GiB", batch_size=8, image_size=32, verify=False)
    after = [(p.device, p.dtype, p.requires_grad, float(p.detach().sum())) for p in model.parameters()]
    assert before == after
    assert model.training is False, "eval mode must be restored"


def test_measurement_allocates_nothing():
    """The whole point: a guard that measures activations must not itself use memory."""
    from torch_preflight.vram.guard import measure_activation_bytes

    model = _conv_model()
    measured = measure_activation_bytes(model, torch.zeros(4, 3, 64, 64, device="meta"))
    assert measured is not None and measured > 0
    # A meta input cannot have been backed by real storage.
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_activations_stay_unknown_without_a_shape():
    """No seq_len or image_size means nothing to build an input from — say unknown."""
    guard = VRAMGuard(_guard_model(), max_vram="8GiB", batch_size=4, verify=False)
    assert guard.activation_source == "unknown"
    assert guard.profile.activation_bytes_per_sample is None


def test_measurement_can_be_switched_off():
    guard = VRAMGuard(_conv_model(), max_vram="8GiB", batch_size=8, image_size=32,
                      measure_activations=False, verify=False)
    assert guard.activation_source == "unknown"
    assert guard.profile.activation_bytes_per_sample is None


def test_an_explicit_activation_figure_wins_over_measurement():
    guard = VRAMGuard(_conv_model(), max_vram="8GiB", batch_size=8, image_size=32,
                      activation_bytes_per_sample=12345, verify=False)
    assert guard.activation_source == "caller"
    assert guard.profile.activation_bytes_per_sample == 12345


def test_a_module_that_cannot_run_on_meta_degrades_to_unknown():
    """Failure must widen the interval, never be mistaken for zero activations."""

    class NeedsRealData(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, x):
            if x.sum().item() > 0:  # .item() cannot work on a meta tensor
                return self.linear(x)
            return self.linear(x)

    guard = VRAMGuard(NeedsRealData(), max_vram="8GiB", batch_size=4, seq_len=8,
                      verify=False)
    assert guard.activation_source == "unknown"
    assert guard.profile.activation_bytes_per_sample is None


def test_an_explicit_example_input_is_used():
    """For models whose input shape cannot be guessed from seq_len or image_size."""
    guard = VRAMGuard(_conv_model(), max_vram="8GiB", batch_size=8,
                      example_input=torch.zeros(2, 3, 32, 32), verify=False)
    assert guard.activation_source == "measured"
    assert guard.profile.activation_bytes_per_sample > 0
