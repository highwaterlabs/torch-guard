# Configuration

`pyproject.toml`:

```toml
[tool.torch-preflight]
select = ["TG001", "TG002", "TG003", "TG005"]   # omit to enable everything
ignore = ["TG004"]
exclude = ["tests/fixtures", "notebooks"]
fail_on = "warning"                              # default: "error"

[tool.torch-preflight.severity]
TG004 = "note"                                   # downgrade instead of disabling
```

A standalone `.torch-preflight.toml` (same keys, no `[tool.torch-preflight]` header) also works.
torch-preflight walks up from the first checked path until it finds one.

## Suppressing findings

```python
losses.append(loss)  # noqa: TG001
losses.append(loss)  # torch-preflight: ignore[TG001]
losses.append(loss)  # noqa
```

`# torch-preflight: skip-file` anywhere in a file skips it entirely.

