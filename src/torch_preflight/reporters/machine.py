"""Machine-readable output: JSON, GitHub workflow annotations, SARIF."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..diagnostics import Diagnostic, Severity
from ..engine import Result
from ..rules import RULES

_GITHUB_LEVEL = {
    Severity.ERROR: "error",
    Severity.WARNING: "warning",
    Severity.NOTE: "notice",
}

_SARIF_LEVEL = {
    Severity.ERROR: "error",
    Severity.WARNING: "warning",
    Severity.NOTE: "note",
}


def _as_dict(diagnostic: Diagnostic) -> Dict[str, Any]:
    return {
        "code": diagnostic.code,
        "message": diagnostic.message,
        "severity": diagnostic.severity.value,
        "category": diagnostic.category.value,
        "path": diagnostic.path,
        "line": diagnostic.line,
        "column": diagnostic.column,
        "end_line": diagnostic.end_line,
        "end_column": diagnostic.end_column,
        "hint": diagnostic.hint,
        "fixable": diagnostic.fixable,
        "fix": diagnostic.fix_summary,
    }


def render_json(result: Result) -> str:
    payload = {
        "version": 1,
        "summary": {
            "files_checked": result.files_checked,
            "errors": result.counts_by_severity()[Severity.ERROR],
            "warnings": result.counts_by_severity()[Severity.WARNING],
            "notes": result.counts_by_severity()[Severity.NOTE],
            "fixed": len(result.fixed),
        },
        "diagnostics": [_as_dict(d) for d in result.diagnostics],
        "failures": [{"path": f.path, "error": f.error} for f in result.errors],
    }
    return json.dumps(payload, indent=2)


def _escape(value: str) -> str:
    """GitHub workflow command escaping."""
    return (
        value.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


def render_github(result: Result) -> str:
    """``::error file=...,line=...::message`` — renders as inline PR annotations."""
    lines: List[str] = []
    for diagnostic in result.diagnostics:
        level = _GITHUB_LEVEL[diagnostic.severity]
        message = diagnostic.message
        if diagnostic.hint:
            message = f"{message}\n\nhelp: {diagnostic.hint}"
        properties = ",".join(
            [
                f"file={_escape(diagnostic.path)}",
                f"line={diagnostic.line}",
                f"endLine={diagnostic.end_line or diagnostic.line}",
                f"col={diagnostic.column}",
                f"title={_escape(f'{diagnostic.code} {RULES[diagnostic.code].summary}')}",
            ]
        )
        lines.append(f"::{level} {properties}::{_escape(message)}")
    return "\n".join(lines)


def render_sarif(result: Result) -> str:
    """SARIF 2.1.0 — consumable by GitHub code scanning and most IDEs."""
    rules = [
        {
            "id": code,
            "name": rule.name,
            "shortDescription": {"text": rule.summary},
            "fullDescription": {"text": rule.explanation},
            "defaultConfiguration": {"level": _SARIF_LEVEL[rule.severity]},
            "properties": {"category": rule.category.value, "tags": [rule.category.value]},
        }
        for code, rule in sorted(RULES.items())
    ]

    results = []
    for diagnostic in result.diagnostics:
        results.append(
            {
                "ruleId": diagnostic.code,
                "level": _SARIF_LEVEL[diagnostic.severity],
                "message": {"text": diagnostic.message},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": diagnostic.path},
                            "region": {
                                "startLine": diagnostic.line,
                                "startColumn": diagnostic.column,
                                "endLine": diagnostic.end_line or diagnostic.line,
                                "endColumn": diagnostic.end_column or diagnostic.column,
                            },
                        }
                    }
                ],
            }
        )

    from .. import __version__

    return json.dumps(
        {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "torch-preflight",
                            "version": __version__,
                            "informationUri": "https://github.com/highwaterlabs/torch-preflight",
                            "rules": rules,
                        }
                    },
                    "results": results,
                }
            ],
        },
        indent=2,
    )
