# How it works

```
source.py
   │
   ├─ LibCST parse ──────────────► concrete syntax tree (comments + formatting preserved)
   │
   ├─ FileContext ───────────────► imports, CUDA usage, Lightning detection, softmax vars
   │
   ├─ Provenance analysis ───────► which names carry a live autograd graph?
   │      seeds: .backward() receivers, criterion/model call results, requires_grad=True
   │      propagate to a fixpoint across assignments
   │      never propagate through .detach() / .item() / float()
   │
   ├─ Rules (scope + loop + no_grad aware) ──► diagnostics [+ optional fix]
   │
   └─ Reporters ─────────────────► terminal · JSON · GitHub annotations · SARIF
```

The provenance pass is the part a syntax-only linter cannot replicate. `losses.append(x)`
is only a leak if `x` is grad-bearing, so torch-preflight tracks where `x` came from —
across assignments, arithmetic, tensor methods and function scopes — and stays quiet when
the value was detached or the code is already inside `torch.no_grad()`.

