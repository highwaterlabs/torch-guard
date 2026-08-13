# Configuration

`pyproject.toml`:

```toml
[tool.torch-guard]
select = ["TG001", "TG002", "TG003", "TG005"]   # omit to enable everything
ignore = ["TG004"]
exclude = ["tests/fixtures", "notebooks"]
fail_on = "warning"                              # default: "error"

[tool.torch-guard.severity]
TG004 = "note"                                   # downgrade instead of disabling
```

A standalone `.torch-guard.toml` (same keys, no `[tool.torch-guard]` header) also works.
torch-guard walks up from the first checked path until it finds one.

## Suppressing findings

```python
losses.append(loss)  # noqa: TG001
losses.append(loss)  # torch-guard: ignore[TG001]
losses.append(loss)  # noqa
```

`# torch-guard: skip-file` anywhere in a file skips it entirely.

