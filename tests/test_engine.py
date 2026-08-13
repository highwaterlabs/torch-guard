"""Suppressions, configuration, discovery and reporters."""

import json
import textwrap

import pytest
from conftest import analyze, codes

from torch_preflight.config import Config, load_config
from torch_preflight.diagnostics import Severity
from torch_preflight.engine import check_file, check_paths, check_source, iter_python_files
from torch_preflight.reporters import render_github, render_json, render_sarif

LEAKY = """
def train(model, loader, criterion, optimizer):
    losses = []
    for batch, y in loader:
        loss = criterion(model(batch), y)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss){suffix}
"""


# ----------------------------------------------------------------- suppression


@pytest.mark.parametrize(
    "suffix",
    [
        "  # noqa",
        "  # noqa: TG001",
        "  # noqa: TG001, TG003",
        "  # torch-preflight: ignore",
        "  # torch-preflight: ignore[TG001]",
    ],
)
def test_suppression_comments(suffix):
    assert codes(LEAKY.format(suffix=suffix)) == []


def test_suppression_is_code_specific():
    assert codes(LEAKY.format(suffix="  # noqa: TG999")) == ["TG001"]


def test_skip_file():
    assert codes("# torch-preflight: skip-file\n" + LEAKY.format(suffix="")) == []


# -------------------------------------------------------------------- config


def test_select_limits_rules():
    cfg = Config(select={"TG003"})
    assert codes(LEAKY.format(suffix=""), cfg) == []


def test_ignore_drops_rules():
    cfg = Config(ignore={"TG001"})
    assert codes(LEAKY.format(suffix=""), cfg) == []


def test_severity_override():
    cfg = Config(severity_overrides={"TG001": Severity.NOTE})
    diagnostics = analyze(LEAKY.format(suffix=""), cfg)
    assert [d.severity for d in diagnostics] == [Severity.NOTE]


def test_load_config_from_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [tool.torch-preflight]
            ignore = ["TG004"]
            fail_on = "warning"

            [tool.torch-preflight.severity]
            TG001 = "note"
            """
        )
    )
    cfg = load_config(tmp_path)
    assert cfg.ignore == {"TG004"}
    assert cfg.fail_on is Severity.WARNING
    assert cfg.severity_overrides == {"TG001": Severity.NOTE}


def test_load_config_from_standalone_file(tmp_path):
    (tmp_path / ".torch-preflight.toml").write_text('select = ["TG001"]\n')
    assert load_config(tmp_path).select == {"TG001"}


def test_load_config_rejects_bad_severity(tmp_path):
    (tmp_path / ".torch-preflight.toml").write_text('[severity]\nTG001 = "critical"\n')
    with pytest.raises(ValueError, match="invalid severity"):
        load_config(tmp_path)


def test_missing_config_is_not_an_error(tmp_path):
    cfg = load_config(tmp_path / "nowhere")
    assert cfg.select is None and cfg.ignore == set()


# ----------------------------------------------------------------- discovery


def test_discovery_skips_excluded_and_vendored_paths(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").write_text("x = 1\n")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "vendored.py").write_text("x = 1\n")
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "scratch.py").write_text("x = 1\n")

    cfg = Config(exclude=["notebooks"])
    found = {p.name for p in iter_python_files([tmp_path], cfg)}
    assert found == {"train.py"}


def test_explicitly_named_file_is_always_checked(tmp_path):
    target = tmp_path / "notebooks" / "scratch.py"
    target.parent.mkdir()
    target.write_text("x = 1\n")
    cfg = Config(exclude=["notebooks"])
    assert iter_python_files([target], cfg) == [target]


# --------------------------------------------------------------- error paths


def test_syntax_error_is_reported_not_raised(tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("def f(:\n")
    result = check_file(broken, Config())
    assert result.error and "syntax error" in result.error
    assert result.diagnostics == []


def test_syntax_error_fails_the_run(tmp_path):
    (tmp_path / "broken.py").write_text("def f(:\n")
    result = check_paths([tmp_path], Config())
    assert result.should_fail(Severity.ERROR)


def test_fail_on_threshold_ignores_warnings_by_default(tmp_path):
    target = tmp_path / "loader.py"
    target.write_text(
        'import torch\nfrom torch.utils.data import DataLoader\n'
        'device = torch.device("cuda")\nloader = DataLoader(ds)\n'
    )
    result = check_paths([tmp_path], Config())
    assert result.counts_by_severity()[Severity.WARNING] == 2
    assert not result.should_fail(Severity.ERROR)
    assert result.should_fail(Severity.WARNING)


# ---------------------------------------------------------------- reporters


def test_json_report_is_valid_and_complete():
    source = textwrap.dedent(LEAKY.format(suffix=""))
    result = check_paths([], Config())
    diagnostics, _ = check_source("t.py", source)
    result.files.append(_FakeFile("t.py", diagnostics))

    payload = json.loads(render_json(result))
    assert payload["summary"]["errors"] == 1
    assert payload["diagnostics"][0]["code"] == "TG001"
    assert payload["diagnostics"][0]["fixable"] is True


def test_sarif_report_is_valid():
    diagnostics, _ = check_source("t.py", textwrap.dedent(LEAKY.format(suffix="")))
    result = check_paths([], Config())
    result.files.append(_FakeFile("t.py", diagnostics))

    payload = json.loads(render_sarif(result))
    assert payload["version"] == "2.1.0"
    driver = payload["runs"][0]["tool"]["driver"]
    from torch_preflight.rules import RULES

    assert {r["id"] for r in driver["rules"]} == set(RULES)
    assert payload["runs"][0]["results"][0]["ruleId"] == "TG001"


def test_github_annotations_escape_special_characters():
    diagnostics, _ = check_source("t.py", textwrap.dedent(LEAKY.format(suffix="")))
    result = check_paths([], Config())
    result.files.append(_FakeFile("t.py", diagnostics))

    output = render_github(result)
    assert output.startswith("::error file=t.py,line=")
    # Commas and colons inside the message must not break the annotation syntax.
    body = output.split("::", 2)[2]
    assert "," not in body.split("%2C")[0] or "%2C" in body


class _FakeFile:
    """Minimal stand-in for engine.FileResult in reporter tests."""

    def __init__(self, path, diagnostics):
        self.path = path
        self.diagnostics = diagnostics
        self.fixed = []
        self.error = None
        self.new_source = None


# ------------------------------------------------------------------ parallelism

_LEAKY_FILE = textwrap.dedent(
    """
    def train(model, loader, criterion, optimizer):
        losses = []
        for batch, y in loader:
            loss = criterion(model(batch), y)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss)
    """
).lstrip("\n")


def _many_files(tmp_path, count):
    from torch_preflight.engine import PARALLEL_THRESHOLD

    assert count >= PARALLEL_THRESHOLD, "need enough files to trigger the parallel path"
    for i in range(count):
        (tmp_path / f"mod_{i:03d}.py").write_text(
            _LEAKY_FILE if i % 3 == 0 else "x = 1\n"
        )
    return tmp_path


def _key(result):
    return sorted((d.path, d.line, d.column, d.code) for d in result.diagnostics)


def test_parallel_and_sequential_agree(tmp_path):
    """Workers must not change what is reported, only how fast it arrives."""
    _many_files(tmp_path, 30)
    sequential = check_paths([tmp_path], Config(), jobs=1)
    parallel = check_paths([tmp_path], Config())

    assert len(parallel.files) == len(sequential.files) == 30
    assert _key(parallel) == _key(sequential)
    assert parallel.counts_by_severity() == sequential.counts_by_severity()


def test_parallel_preserves_fixability_labels(tmp_path):
    """Fix objects cannot be pickled; the label that drives the UI must survive."""
    _many_files(tmp_path, 30)
    parallel = check_paths([tmp_path], Config())

    fixable = [d for d in parallel.diagnostics if d.fixable]
    assert fixable, "expected TG001 findings to advertise a fix"
    for diagnostic in fixable:
        assert diagnostic.fix is None          # stripped for transport
        assert diagnostic.fix_summary          # but still reportable


def test_fix_mode_stays_sequential_and_rewrites(tmp_path):
    """--fix needs the live tree the rules ran against, so it must not go parallel."""
    _many_files(tmp_path, 30)
    result = check_paths([tmp_path], Config(), fix=True, write=True)

    assert result.fixed, "expected fixes to be applied"
    assert "loss.detach()" in (tmp_path / "mod_000.py").read_text()


def test_positions_are_resolved_for_every_finding(tmp_path):
    """Positions are filled in lazily; a finding with line 0 would be a regression."""
    _many_files(tmp_path, 30)
    for result in (check_paths([tmp_path], Config(), jobs=1),
                   check_paths([tmp_path], Config())):
        assert result.diagnostics
        for diagnostic in result.diagnostics:
            assert diagnostic.line > 0 and diagnostic.column > 0
            assert diagnostic.node is None
