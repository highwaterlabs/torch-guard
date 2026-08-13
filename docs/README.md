# torch-preflight documentation

Start with the [README](../README.md) if you just want to install it and run it. These
pages are the detail behind it.

| Page | What's in it |
|---|---|
| [rules.md](rules.md) | Every rule, what it costs you, and the false positives we deliberately suppress |
| [vram-estimation.md](vram-estimation.md) | Pre-flight estimation, custom architectures, the CI gate, `VRAMGuard`, and how accurate any of it is |
| [cli.md](cli.md) | Every command and flag, exit codes, autofixes |
| [configuration.md](configuration.md) | `pyproject.toml` settings and inline suppression |
| [ci.md](ci.md) | GitHub Action, pre-commit, SARIF into code scanning |
| [architecture.md](architecture.md) | How the analysis pipeline fits together |
| [development.md](development.md) | Running the tests, adding a rule, what's next |

Design decisions and the reasoning behind the numbers live in [design/](../design/) —
[RFC 0001](../design/rfcs/0001-vram-estimator.md) covers the estimator, and
[spike 0001](../design/spikes/0001-meta-device-activation-capture.md) is the measurement
work the cost model rests on.
