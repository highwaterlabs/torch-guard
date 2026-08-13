"""Configuration loading from ``pyproject.toml`` / ``.torch-guard.toml``."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from .diagnostics import Severity

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.9/3.10
    import tomli as tomllib

DEFAULT_EXCLUDES = [
    ".git", ".hg", ".svn", ".tox", ".nox", ".venv", "venv", "env",
    "__pycache__", "node_modules", "build", "dist", ".eggs", "site-packages",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints",
]

CONFIG_FILENAMES = (".torch-guard.toml", "torch-guard.toml", "pyproject.toml")


@dataclass
class Config:
    select: Optional[Set[str]] = None
    ignore: Set[str] = field(default_factory=set)
    exclude: List[str] = field(default_factory=list)
    severity_overrides: Dict[str, Severity] = field(default_factory=dict)
    fail_on: Severity = Severity.ERROR
    #: Target hardware for TG010's projected-OOM gate. Unset means TG010 stays silent.
    target_gpu: Optional[str] = None
    source: Optional[str] = None

    def is_enabled(self, code: str) -> bool:
        if code in self.ignore:
            return False
        return self.select is None or code in self.select

    def severity_for(self, code: str, default: Severity) -> Severity:
        return self.severity_overrides.get(code, default)

    def is_excluded(self, path: Path) -> bool:
        parts = path.parts
        if any(part in DEFAULT_EXCLUDES for part in parts):
            return True
        text = path.as_posix()
        for pattern in self.exclude:
            pattern = pattern.rstrip("/")
            if fnmatch(text, pattern) or fnmatch(text, f"{pattern}/*") or f"/{pattern}/" in f"/{text}/":
                return True
        return False


def _coerce(data: dict, source: str) -> Config:
    cfg = Config(source=source)

    select = data.get("select")
    if select:
        cfg.select = {str(code).upper() for code in select}

    cfg.ignore = {str(code).upper() for code in data.get("ignore", [])}
    cfg.exclude = [str(p) for p in data.get("exclude", [])]

    for code, level in (data.get("severity") or {}).items():
        try:
            cfg.severity_overrides[str(code).upper()] = Severity(str(level).lower())
        except ValueError as exc:
            raise ValueError(
                f"{source}: invalid severity {level!r} for {code} "
                f"(expected error, warning or note)"
            ) from exc

    if data.get("target_gpu"):
        cfg.target_gpu = str(data["target_gpu"])

    if "fail_on" in data:
        try:
            cfg.fail_on = Severity(str(data["fail_on"]).lower())
        except ValueError as exc:
            raise ValueError(f"{source}: invalid fail_on {data['fail_on']!r}") from exc

    return cfg


def load_config(start: Path, explicit: Optional[Path] = None) -> Config:
    """Find and load configuration, walking up from ``start`` to the filesystem root."""
    if explicit is not None:
        data = tomllib.loads(explicit.read_text(encoding="utf-8"))
        if explicit.name == "pyproject.toml":
            data = data.get("tool", {}).get("torch-guard", {})
        return _coerce(data, str(explicit))

    start = start if start.is_dir() else start.parent
    for directory in [start, *start.parents]:
        for name in CONFIG_FILENAMES:
            candidate = directory / name
            if not candidate.is_file():
                continue
            data = tomllib.loads(candidate.read_text(encoding="utf-8"))
            if name == "pyproject.toml":
                data = data.get("tool", {}).get("torch-guard")
                if data is None:
                    continue
            return _coerce(data, str(candidate))
    return Config()


def apply_cli_overrides(
    cfg: Config,
    *,
    select: Optional[Sequence[str]] = None,
    ignore: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    fail_on: Optional[str] = None,
    target_gpu: Optional[str] = None,
) -> Config:
    """CLI flags win over file configuration."""
    if select:
        cfg.select = {code.upper() for code in select}
    if ignore:
        cfg.ignore |= {code.upper() for code in ignore}
    if exclude:
        cfg.exclude = [*cfg.exclude, *exclude]
    if fail_on:
        cfg.fail_on = Severity(fail_on.lower())
    if target_gpu:
        cfg.target_gpu = target_gpu
    return cfg
