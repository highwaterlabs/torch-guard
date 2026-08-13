import textwrap
from typing import List, Optional

import pytest

from torch_guard.config import Config
from torch_guard.diagnostics import Diagnostic
from torch_guard.engine import check_source


def analyze(source: str, cfg: Optional[Config] = None) -> List[Diagnostic]:
    """Run every rule over an inline snippet."""
    diagnostics, _ = check_source("t.py", textwrap.dedent(source).lstrip("\n"), cfg)
    return diagnostics


def codes(source: str, cfg: Optional[Config] = None) -> List[str]:
    return sorted(d.code for d in analyze(source, cfg))


@pytest.fixture
def check():
    return codes
