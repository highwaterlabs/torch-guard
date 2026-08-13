"""Autofix correctness: fixes must be exact and must not disturb anything else."""

import textwrap

from torch_guard.config import Config
from torch_guard.engine import check_file, check_paths, check_source
from torch_guard.fixer import apply_fixes


def fixed_source(source: str) -> str:
    source = textwrap.dedent(source).lstrip("\n")
    diagnostics, ctx = check_source("t.py", source)
    new_source, applied = apply_fixes(ctx.module, diagnostics)
    assert applied, "expected at least one fix to apply"
    return new_source


def test_tg001_appends_detach():
    out = fixed_source(
        """
        def train(model, loader, criterion, optimizer):
            losses = []
            for batch, y in loader:
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss)
        """
    )
    assert "losses.append(loss.detach())" in out


def test_tg001_parenthesises_compound_expressions():
    out = fixed_source(
        """
        def train(model, loader, criterion, optimizer):
            losses = []
            for batch, y in loader:
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss * 2 + 1)
        """
    )
    assert "losses.append((loss * 2 + 1).detach())" in out


def test_tg002_inserts_decorator_above_existing_ones():
    out = fixed_source(
        """
        import torch

        @staticmethod
        def validate(model, loader):
            for x, y in loader:
                out = model(x)
        """
    )
    assert "@torch.no_grad()\n@staticmethod\ndef validate" in out


def test_tg004_adds_pin_memory():
    out = fixed_source(
        """
        import torch
        from torch.utils.data import DataLoader

        device = torch.device("cuda")
        loader = DataLoader(ds, num_workers=8)
        """
    )
    assert "DataLoader(ds, num_workers=8, pin_memory=True)" in out


def test_tg005_unwraps_redundant_softmax():
    out = fixed_source(
        """
        import torch.nn.functional as F

        def step(model, x, y):
            logits = model(x)
            return F.cross_entropy(F.softmax(logits, dim=1), y)
        """
    )
    assert "F.cross_entropy(logits, y)" in out


def test_fixes_preserve_comments_and_formatting():
    source = textwrap.dedent(
        """
        def train(model, loader, criterion, optimizer):
            # keep me
            losses = []


            for batch, y in loader:    # and me
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss)  # trailing note
        """
    ).lstrip("\n")

    diagnostics, ctx = check_source("t.py", source)
    new_source, _ = apply_fixes(ctx.module, diagnostics)

    assert "# keep me" in new_source
    assert "# and me" in new_source
    assert "losses.append(loss.detach())  # trailing note" in new_source
    # Only the targeted line changed.
    changed = [
        (a, b)
        for a, b in zip(source.splitlines(), new_source.splitlines())
        if a != b
    ]
    assert len(changed) == 1


def test_fixed_output_is_clean_on_reanalysis():
    """Applying fixes must actually resolve the findings, not shuffle them."""
    out = fixed_source(
        """
        import torch
        import torch.nn.functional as F

        def validate(model, loader):
            for x, y in loader:
                logits = model(x)
                loss = F.cross_entropy(F.softmax(logits, dim=1), y)
        """
    )
    remaining, _ = check_source("t.py", out)
    assert [d.code for d in remaining] == []


def test_check_file_rewrites_only_with_fix_enabled(tmp_path):
    target = tmp_path / "train.py"
    original = textwrap.dedent(
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
    target.write_text(original)

    result = check_file(target, Config(), fix=False)
    assert result.new_source is None
    assert target.read_text() == original

    check_paths([tmp_path], Config(), fix=True, write=True)
    assert "loss.detach()" in target.read_text()


def test_fixed_diagnostics_are_removed_from_the_report(tmp_path):
    target = tmp_path / "train.py"
    target.write_text(
        textwrap.dedent(
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
    )
    result = check_paths([tmp_path], Config(), fix=True, write=False)
    assert [d.code for d in result.fixed] == ["TG001"]
    assert result.diagnostics == []
