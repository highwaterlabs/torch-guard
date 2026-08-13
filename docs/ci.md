# CI integration

## GitHub Action

```yaml
- uses: highwaterlabs/torch-preflight@v0
  with:
    paths: src/
    format: github        # inline PR annotations
```

Or with code scanning:

```yaml
- uses: highwaterlabs/torch-preflight@v0
  with:
    paths: src/
    format: sarif
    output: torch-preflight.sarif
- uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: torch-preflight.sarif
```

## Pre-commit

```yaml
repos:
  - repo: https://github.com/highwaterlabs/torch-preflight
    rev: v0.1.0
    hooks:
      - id: torch-preflight
```

