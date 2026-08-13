"""Meta-device provider and model autodetection. Requires the ``[vram]`` extra.

Split out of test_vram.py deliberately: these tests cannot even be *collected* without
torch, because their parametrize decorators reference ``torch`` dtypes. Keeping them in
the same module as the dependency-free tests made a bare ``pip install -e .[dev]`` skip
the entire file — CI was green while silently not running any of it.
"""

import textwrap

import pytest

torch = pytest.importorskip("torch", reason="meta provider needs the [vram] extra")

from torch_guard.analysis.context import build_context  # noqa: E402
from torch_guard.vram import autodetect as autodetect_mod  # noqa: E402
from torch_guard.vram.costmodel import params_from_transformer_shape  # noqa: E402
from torch_guard.vram.providers import meta, resolve_profile  # noqa: E402
from torch_guard.vram.types import (  # noqa: E402
    Confidence,
    PrecisionMode,
    RiskBand,
    RunConfig,
    TransformerShape,
)
from meta_fixture_model import Tiny  # noqa: E402

assert hasattr(autodetect_mod, "module_is_import_safe"), "submodule shadowed by a function"


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


def test_unimportable_module_is_named_in_the_reason():
    profile = resolve_profile("mypkg.models:build_gpt")
    assert profile.confidence is Confidence.UNKNOWN
    assert "could not import" in profile.reason


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


