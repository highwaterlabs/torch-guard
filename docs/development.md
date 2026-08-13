# Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Adding a rule: create `src/torch_preflight/rules/tg0NN_name.py`, subclass `Rule`, decorate with
`@register`, import it in `rules/__init__.py`, then add cases to `tests/test_rules.py` —
at least one that must fire and one that must stay quiet.

# Roadmap

The rule engine is designed so a new check is one file plus a `@register` decorator.
Next up:

- **TG006** `nn.BCELoss` on raw logits (should be `BCEWithLogitsLoss`)
- **TG007** CPU↔GPU thrashing (`.cpu().numpy()`) inside the training loop
- **TG008** non-reproducible runs — `torch.rand` without seeding `torch`, `numpy` and `random`
- **TG009** in-place ops on tensors needed for the backward pass
- **Meta-device profiling** — exact parameter and activation measurement for arbitrary
  custom models via PyTorch's `meta` device, as the `[vram]` extra ([RFC 0001](../design/rfcs/0001-vram-estimator.md)
  phase 2, gated on [spike 0001](../design/spikes/0001-meta-device-activation-capture.md))
- **Model autodetection** — resolve locally defined model classes by constant-folding their
  constructor arguments

