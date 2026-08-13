"""Human-readable terminal output."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.text import Text

from ..diagnostics import Category, Diagnostic, Severity
from ..engine import Result

SEVERITY_STYLE = {
    Severity.ERROR: "bold red",
    Severity.WARNING: "bold yellow",
    Severity.NOTE: "bold cyan",
}

CATEGORY_STYLE = {
    Category.CRITICAL_OOM: "red",
    Category.CONVERGENCE_BUG: "magenta",
    Category.PERFORMANCE_WARN: "yellow",
}


def _source_lines(path: str) -> Optional[List[str]]:
    try:
        return Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None


def _render_snippet(console: Console, diagnostic: Diagnostic, lines: List[str]) -> None:
    index = diagnostic.line - 1
    if not (0 <= index < len(lines)):
        return

    raw = lines[index]
    gutter = f"{diagnostic.line:>5} "
    console.print(
        Text(gutter, style="dim") + Text("│ ", style="dim") + Text(raw.rstrip())
    )

    # Underline the offending span on single-line findings.
    if diagnostic.end_line in (None, diagnostic.line):
        start = max(diagnostic.column - 1, 0)
        end = diagnostic.end_column - 1 if diagnostic.end_column else start + 1
        width = max(end - start, 1)
        # Preserve tabs so the caret lines up with the rendered source.
        prefix = "".join("\t" if ch == "\t" else " " for ch in raw[:start])
        caret = Text(" " * 5 + " │ ", style="dim") + Text(
            prefix + "^" * width, style=SEVERITY_STYLE[diagnostic.severity]
        )
        console.print(caret)


def render(result: Result, console: Optional[Console] = None, *, show_source: bool = True) -> None:
    console = console or Console()

    by_file: Dict[str, List[Diagnostic]] = {}
    for diagnostic in result.diagnostics:
        by_file.setdefault(diagnostic.path, []).append(diagnostic)

    for path, diagnostics in by_file.items():
        console.print()
        console.print(Text(path, style="bold underline"))
        lines = _source_lines(path) if show_source else None

        for index, diagnostic in enumerate(diagnostics):
            if index:
                console.print()
            location = Text(f"{diagnostic.line}:{diagnostic.column}", style="dim")
            severity = Text(
                f"{diagnostic.severity.value:<8}", style=SEVERITY_STYLE[diagnostic.severity]
            )
            code = Text(diagnostic.code, style="bold")
            category = Text(
                f"({diagnostic.category.value})", style=CATEGORY_STYLE[diagnostic.category]
            )
            console.print(
                Text("  ") + location + Text("  ") + severity + code + Text(" ") + category
            )
            console.print(Text("  " + diagnostic.message))

            if lines:
                _render_snippet(console, diagnostic, lines)

            if diagnostic.hint:
                console.print(Text("  help: ", style="green") + Text(diagnostic.hint))
            if diagnostic.fixable:
                console.print(
                    Text("  fix:  ", style="cyan")
                    + Text(f"{diagnostic.fix_summary} (run with --fix)")
                )

    _render_summary(result, console)


def _render_summary(result: Result, console: Console) -> None:
    counts = result.counts_by_severity()
    total = sum(counts.values())

    console.print()
    if result.errors:
        for file in result.errors:
            console.print(Text(f"  {file.path}: {file.error}", style="red"))
        console.print()

    if total == 0 and not result.errors:
        console.print(
            Text("✔ ", style="bold green")
            + Text(f"No issues found in {result.files_checked} file(s).")
        )
        return

    parts = Text()
    for severity in (Severity.ERROR, Severity.WARNING, Severity.NOTE):
        if counts[severity]:
            if parts:
                parts.append(", ")
            parts.append(str(counts[severity]), style=SEVERITY_STYLE[severity])
            parts.append(f" {severity.value}{'s' if counts[severity] != 1 else ''}")

    console.print(
        Text("Found ") + parts + Text(f" in {result.files_checked} file(s).")
    )

    fixable = sum(1 for d in result.diagnostics if d.fixable)
    if fixable:
        console.print(
            Text(f"{fixable} issue(s) can be fixed automatically with ", style="cyan")
            + Text("--fix", style="bold cyan")
            + Text(".", style="cyan")
        )
    if result.fixed:
        console.print(Text(f"Fixed {len(result.fixed)} issue(s).", style="green"))
