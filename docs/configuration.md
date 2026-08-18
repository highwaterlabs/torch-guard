# Configuration

`pyproject.toml`:

```toml
[tool.torch-preflight]
select = ["TG001", "TG002", "TG003", "TG005"]   # omit to enable everything
ignore = ["TG004"]
exclude = ["tests/fixtures", "notebooks"]
fail_on = "warning"                              # default: "error"

[tool.torch-preflight.severity]
TG001 = "error"                                  # escalate: our jobs run 500k steps
TG008 = "note"                                   # downgrade instead of disabling
```

`fail_on` sets the gate; `severity` re-levels individual rules and is the escape hatch when a
default does not suit your run lengths. `TG001 = "error"` is the worth-knowing one: a graph
retained *after* its backward leaks about 15 KiB per iteration of host memory, which is
nothing over 5,000 steps and 1.4 GiB over 100,000 — so we ship it as a warning and let anyone
running long jobs raise it. See [RFC 0003](../design/rfcs/0003-severity-and-ci-gating.md) for
what the three levels mean and how they were measured.

A standalone `.torch-preflight.toml` (same keys, no `[tool.torch-preflight]` header) also works.
torch-preflight walks up from the first checked path until it finds one.

## Suppressing findings

```python
losses.append(loss)  # noqa: TG001
losses.append(loss)  # torch-preflight: ignore[TG001]
losses.append(loss)  # noqa
```

`# torch-preflight: skip-file` anywhere in a file skips it entirely.

