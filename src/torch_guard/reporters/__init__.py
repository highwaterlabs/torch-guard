"""Output formats."""

from .machine import render_github, render_json, render_sarif
from .terminal import render as render_terminal

FORMATS = ("terminal", "json", "github", "sarif")

__all__ = ["FORMATS", "render_github", "render_json", "render_sarif", "render_terminal"]
