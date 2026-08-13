"""File discovery, rule execution and suppression handling."""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import libcst as cst
from libcst.metadata import PositionProvider

from .analysis.context import FileContext, build_context
from .config import Config
from .diagnostics import Diagnostic, Severity
from .fixer import apply_fixes
from .rules import RuleDispatcher, all_rules

#: ``# noqa``, ``# noqa: TG001,TG002``, ``# torch-preflight: ignore[TG003]``
_SUPPRESS_RE = re.compile(
    r"#\s*(?:noqa|torch-preflight\s*:\s*ignore)(?:\s*[:\[]\s*(?P<codes>[A-Za-z0-9,\s]+?)\s*\]?)?"
    r"(?:\s|$|#)",
    re.IGNORECASE,
)
_SKIP_FILE_RE = re.compile(r"#\s*torch-preflight\s*:\s*skip[-_]file", re.IGNORECASE)

_ALL = "*"


@dataclass
class FileResult:
    path: str
    diagnostics: List[Diagnostic] = field(default_factory=list)
    fixed: List[Diagnostic] = field(default_factory=list)
    error: Optional[str] = None
    #: Set when ``fix=True`` and the rewritten source differs from the original.
    new_source: Optional[str] = None


@dataclass
class Result:
    files: List[FileResult] = field(default_factory=list)

    @property
    def diagnostics(self) -> List[Diagnostic]:
        out: List[Diagnostic] = []
        for file in self.files:
            out.extend(file.diagnostics)
        return sorted(out, key=lambda d: d.sort_key())

    @property
    def fixed(self) -> List[Diagnostic]:
        return [d for file in self.files for d in file.fixed]

    @property
    def errors(self) -> List[FileResult]:
        return [f for f in self.files if f.error]

    @property
    def files_checked(self) -> int:
        return len([f for f in self.files if not f.error])

    def counts_by_severity(self) -> Dict[Severity, int]:
        counts = {severity: 0 for severity in Severity}
        for diagnostic in self.diagnostics:
            counts[diagnostic.severity] += 1
        return counts

    def should_fail(self, threshold: Severity) -> bool:
        if self.errors:
            return True
        return any(d.severity.rank >= threshold.rank for d in self.diagnostics)


# --------------------------------------------------------------------- discovery


def iter_python_files(paths: Sequence[Path], cfg: Config) -> List[Path]:
    """Expand ``paths`` into a sorted list of .py files, honouring excludes."""
    found: List[Path] = []
    seen: Set[Path] = set()

    for raw in paths:
        path = Path(raw)
        if path.is_file():
            # An explicitly named file is checked even if it matches an exclude.
            if path.suffix in (".py", ".pyi") and path not in seen:
                seen.add(path)
                found.append(path)
            continue
        for candidate in sorted(path.rglob("*.py")):
            if candidate in seen or cfg.is_excluded(candidate):
                continue
            seen.add(candidate)
            found.append(candidate)

    return found


# ---------------------------------------------------------------- suppressions


def _suppressions(source: str) -> Tuple[bool, Dict[int, Set[str]]]:
    """Parse suppression comments. Returns ``(skip_whole_file, {line: {codes}})``."""
    if _SKIP_FILE_RE.search(source):
        return True, {}

    per_line: Dict[int, Set[str]] = {}
    for number, line in enumerate(source.splitlines(), start=1):
        if "#" not in line:
            continue
        match = _SUPPRESS_RE.search(line)
        if match is None:
            continue
        codes = match.group("codes")
        if codes:
            per_line[number] = {c.strip().upper() for c in codes.split(",") if c.strip()}
        else:
            per_line[number] = {_ALL}
    return False, per_line


def _is_suppressed(diagnostic: Diagnostic, per_line: Dict[int, Set[str]]) -> bool:
    lines = {diagnostic.line}
    if diagnostic.end_line:
        lines.update(range(diagnostic.line, diagnostic.end_line + 1))
    for line in lines:
        codes = per_line.get(line)
        if codes and (_ALL in codes or diagnostic.code in codes):
            return True
    return False


# ------------------------------------------------------------------- execution


def check_source(
    path: str, source: str, cfg: Optional[Config] = None
) -> Tuple[List[Diagnostic], Optional[FileContext]]:
    """Run every enabled rule over one file's source. Raises nothing on lint errors."""
    cfg = cfg or Config()

    skip_file, per_line = _suppressions(source)
    if skip_file:
        return [], None

    ctx = build_context(path, source)

    rules = []
    for rule_cls in all_rules():
        if not cfg.is_enabled(rule_cls.code):
            continue
        if not rule_cls.should_run(ctx, cfg):
            continue
        rule = rule_cls(ctx)
        rule.cfg = cfg
        rules.append(rule)

    diagnostics: List[Diagnostic] = []
    if rules:
        # One traversal for every rule. Six separate walks used to be 78% of runtime.
        # Walk the module directly rather than through the MetadataWrapper: that would
        # force PositionProvider resolution, which costs more than parsing the file.
        ctx.module.visit(RuleDispatcher(rules))

        if any(rule.diagnostics for rule in rules):
            positions = ctx.wrapper.resolve(PositionProvider)
            for rule in rules:
                for diagnostic in rule.diagnostics:
                    diagnostic.resolve_position(positions)

        for rule in rules:
            for diagnostic in rule.diagnostics:
                diagnostic.severity = cfg.severity_for(
                    diagnostic.code, diagnostic.severity
                )
                if not _is_suppressed(diagnostic, per_line):
                    diagnostics.append(diagnostic)

    diagnostics.sort(key=lambda d: d.sort_key())
    return diagnostics, ctx


def check_file(path: Path, cfg: Config, *, fix: bool = False) -> FileResult:
    result = FileResult(path=str(path))
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        result.error = f"could not read: {exc}"
        return result

    try:
        diagnostics, ctx = check_source(str(path), source, cfg)
    except cst.ParserSyntaxError as exc:
        result.error = f"syntax error: {exc}"
        return result
    except RecursionError:
        result.error = "file too deeply nested to analyze"
        return result

    result.diagnostics = diagnostics

    if fix and ctx is not None:
        new_source, applied = apply_fixes(ctx.module, diagnostics)
        if applied and new_source != source:
            result.new_source = new_source
            result.fixed = applied
            # Fixed findings are no longer outstanding.
            fixed_ids = {id(d) for d in applied}
            result.diagnostics = [d for d in diagnostics if id(d) not in fixed_ids]

    return result


#: Below this many files, spawning workers costs more than it saves.
PARALLEL_THRESHOLD = 24


def _check_one(task) -> FileResult:
    """Worker entry point. Must be module level so ``spawn`` can import it."""
    path, cfg = task
    file_result = check_file(Path(path), cfg, fix=False)
    for diagnostic in file_result.diagnostics:
        # Fixes hold CST nodes and closures; neither survives pickling. ``--fix`` runs
        # single-process, so only the human-readable label is needed here.
        if diagnostic.fix is not None:
            diagnostic.fix_label = diagnostic.fix.description
            diagnostic.fix = None
        diagnostic.node = None
    return file_result


def check_paths(
    paths: Sequence[Path],
    cfg: Config,
    *,
    fix: bool = False,
    write: bool = True,
    jobs: Optional[int] = None,
) -> Result:
    """Check every file under ``paths``; optionally rewrite them in place.

    Runs across processes when there is enough work to justify it. Applying fixes always
    runs sequentially: the fixer replaces nodes by identity in the tree the rules ran
    against, which cannot be shipped to another process.
    """
    files = iter_python_files(paths, cfg)
    result = Result()

    workers = jobs if jobs is not None else (os.cpu_count() or 1)
    parallel = not fix and workers > 1 and len(files) >= PARALLEL_THRESHOLD

    if parallel:
        try:
            chunk = max(1, len(files) // (workers * 4))
            with ProcessPoolExecutor(max_workers=workers) as pool:
                result.files.extend(
                    pool.map(_check_one, [(str(f), cfg) for f in files], chunksize=chunk)
                )
            return result
        except Exception:
            # Sandboxes and restricted environments can refuse to fork. Falling back is
            # better than failing a lint run over a scheduling detail.
            result = Result()

    for path in files:
        file_result = check_file(path, cfg, fix=fix)
        if fix and write and file_result.new_source is not None:
            path.write_text(file_result.new_source, encoding="utf-8")
        result.files.append(file_result)
    return result
