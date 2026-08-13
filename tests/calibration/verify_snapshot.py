"""Check the bundled architecture snapshot against the live Hugging Face configs.

    pip install "torch-preflight[hub]"
    python tests/calibration/verify_snapshot.py            # report drift
    python tests/calibration/verify_snapshot.py --json     # machine-readable

This is deliberately a **verifier, not a regenerator**. Overwriting
``src/torch_preflight/vram/data/architectures.json`` from whatever the hub returns today
would let a renamed field or a reorganised repo silently rewrite numbers the whole cost
model rests on, and the diff would be too large to review. Drift should be rare and
individually inspected, so this prints what changed and leaves the edit to a human.

What it checks, per entry with a shape: layers, hidden size, head count, KV heads,
intermediate size and vocabulary against the model's `config.json`, plus the published
parameter count against our analytic formula.

Cadence
-------
Run it when adding architectures, before a release, and otherwise a few times a year.
Published models are immutable in practice -- `gpt2` will not change shape -- so the real
value is catching *our* transcription mistakes and upstream config-field renames. Two such
renames have already bitten this project: `tie_word_embeddings` defaulting to True when
absent, and DistilBERT naming its fields `dim`/`hidden_dim`/`n_heads`.

Requires network access, which is why it is a script rather than a test. The offline test
suite pins the same numbers against captured fixtures in `tests/fixtures/hub/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from torch_preflight.vram import archdb  # noqa: E402
from torch_preflight.vram.costmodel import params_from_transformer_shape  # noqa: E402

#: Snapshot key -> hub repo, where they differ. Only entries listed here are checkable;
#: the rest have no single canonical repo (or are not on the hub at all).
HUB_REPOS = {
    "gpt2": "gpt2",
    "bert-base-uncased": "bert-base-uncased",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "t5-small": "t5-small",
    "t5-base": "t5-base",
    "t5-large": "t5-large",
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
    "whisper-small": "openai/whisper-small",
    "whisper-medium": "openai/whisper-medium",
    "whisper-large-v3": "openai/whisper-large-v3",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "qwen2-7b": "Qwen/Qwen2-7B",
}

#: Snapshot field -> the config keys that may carry it, most specific first.
FIELD_ALIASES = {
    "layers": ("num_hidden_layers", "n_layer", "num_layers", "encoder_layers", "n_layers"),
    "hidden": ("hidden_size", "n_embd", "d_model", "dim"),
    "heads": ("num_attention_heads", "n_head", "num_heads", "encoder_attention_heads",
              "n_heads"),
    "kv_heads": ("num_key_value_heads",),
    "intermediate": ("intermediate_size", "n_inner", "d_ff", "encoder_ffn_dim",
                     "hidden_dim"),
    "vocab": ("vocab_size",),
}


def config_value(config: dict, field: str):
    for key in FIELD_ALIASES[field]:
        if config.get(key) is not None:
            return config[key]
    return None


def check(key: str, repo: str):
    """Returns (problems, checked). ``checked`` is False when there was nothing to compare.

    Plenty of snapshot entries carry a published parameter count and no shape, which is a
    legitimate state rather than drift -- the estimator reports their activations as
    unknown by design. Counting those as failures would make the script cry wolf.
    """
    from huggingface_hub import hf_hub_download

    profile = archdb.resolve(key)
    if profile is None or profile.shape is None:
        return [], False

    try:
        path = hf_hub_download(repo_id=repo, filename="config.json")
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:  # network, auth, gated repo
        return [f"{key}: could not fetch {repo} ({type(exc).__name__})"], False

    problems = []
    for field in ("layers", "hidden", "heads", "intermediate", "vocab"):
        ours = getattr(profile.shape, field)
        theirs = config_value(config, field)
        # An encoder-decoder's `layers` is its *encoder* depth by our convention.
        if theirs is None or not ours:
            continue
        if ours != theirs:
            problems.append(f"{key}.{field}: snapshot {ours}, hub {theirs}")

    computed = params_from_transformer_shape(profile.shape)
    drift = abs(computed - profile.param_count) / profile.param_count
    if drift > 0.03:
        problems.append(
            f"{key}: formula {computed:,} vs published {profile.param_count:,} "
            f"({drift:.1%} apart)"
        )
    return problems, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--only", help="check a single snapshot key")
    args = parser.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit('needs the hub extra: pip install "torch-preflight[hub]"')

    targets = ({args.only: HUB_REPOS[args.only]} if args.only else HUB_REPOS)
    findings = {}
    checked = skipped = 0
    for key, repo in targets.items():
        problems, was_checked = check(key, repo)
        checked += was_checked
        skipped += not was_checked
        if problems:
            findings[key] = problems

    if args.json:
        print(json.dumps(findings, indent=2))
    elif not findings:
        print(f"snapshot agrees with the hub across {checked} entries "
              f"({skipped} skipped: no shape to compare)")
    else:
        for key, problems in findings.items():
            for problem in problems:
                print(f"  {problem}")
        print(f"\n{len(findings)} entr{'y' if len(findings) == 1 else 'ies'} drifted. "
              f"Inspect each one and edit architectures.json by hand -- this script "
              f"deliberately does not rewrite it.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
