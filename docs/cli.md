# CLI reference

```bash
torch-preflight check ./src/              # check a tree
torch-preflight check train.py --fix      # apply autofixes in place
torch-preflight check train.py --diff     # show what --fix would do
torch-preflight check . -f json           # machine-readable
torch-preflight check . -f sarif          # GitHub code scanning
torch-preflight check . -f github         # inline PR annotations
torch-preflight rules                     # list rules
torch-preflight explain TG003             # full write-up for one rule
```

`torch-preflight ./src` is shorthand for `torch-preflight check ./src`.

**Speed.** PyTorch's own source — 2285 files — takes about 50 seconds on an 8-core laptop.
Checking runs across processes by default; `--jobs 1` disables that, and `--fix` always
runs sequentially.

**Exit codes:** `0` clean · `1` findings at or above `--fail-on` (default `error`, so
performance warnings do not break CI) · `2` bad invocation.

## Autofixes

Only fixes that cannot change semantics are offered:

| Rule | Fix |
|------|-----|
| TG001 | append `.detach()` to the stored expression |
| TG002 | add `@torch.no_grad()` — **only** to functions whose name says they evaluate, never to a training routine that happens to contain a validation loop |
| TG004 | add `pin_memory=True` |
| TG005 | unwrap the redundant `softmax(...)` call |

TG003 has no autofix: where `zero_grad()` belongs depends on whether you meant to
accumulate gradients, and guessing would be worse than a comment.

Because fixes are LibCST node replacements, everything untouched round-trips byte for
byte — no reformatting, no lost comments.

