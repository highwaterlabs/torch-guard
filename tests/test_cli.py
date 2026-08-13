"""End-to-end CLI behaviour, including exit codes CI depends on."""

import json
import textwrap

from torch_guard.cli import EXIT_ERROR, EXIT_FINDINGS, EXIT_OK, main

LEAKY = textwrap.dedent(
    """
    def train(model, loader, criterion, optimizer):
        losses = []
        for batch, y in loader:
            loss = criterion(model(batch), y)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss)
    """
).lstrip("\n")


def write(tmp_path, name="train.py", source=LEAKY):
    target = tmp_path / name
    target.write_text(source)
    return target


def test_exit_code_on_findings(tmp_path):
    assert main(["check", str(write(tmp_path))]) == EXIT_FINDINGS


def test_exit_code_when_clean(tmp_path):
    assert main(["check", str(write(tmp_path, source="x = 1\n"))]) == EXIT_OK


def test_exit_code_on_bad_path():
    assert main(["check", "/does/not/exist"]) == EXIT_ERROR


def test_path_shorthand_implies_check(tmp_path):
    assert main([str(write(tmp_path))]) == EXIT_FINDINGS


def test_warnings_do_not_fail_by_default(tmp_path):
    source = (
        'import torch\nfrom torch.utils.data import DataLoader\n'
        'device = torch.device("cuda")\nloader = DataLoader(ds)\n'
    )
    target = write(tmp_path, "loader.py", source)
    assert main(["check", str(target)]) == EXIT_OK
    assert main(["check", str(target), "--fail-on", "warning"]) == EXIT_FINDINGS


def test_json_format(tmp_path, capsys):
    main(["check", str(write(tmp_path)), "-f", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["diagnostics"][0]["code"] == "TG001"
    assert payload["summary"]["files_checked"] == 1


def test_github_format(tmp_path, capsys):
    main(["check", str(write(tmp_path)), "-f", "github"])
    assert capsys.readouterr().out.startswith("::error file=")


def test_sarif_format(tmp_path, capsys):
    main(["check", str(write(tmp_path)), "-f", "sarif"])
    assert json.loads(capsys.readouterr().out)["version"] == "2.1.0"


def test_select_and_ignore_flags(tmp_path, capsys):
    target = write(tmp_path)
    main(["check", str(target), "--ignore", "TG001", "-f", "json"])
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []

    main(["check", str(target), "--select", "TG003,TG004", "-f", "json"])
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []


def test_diff_does_not_write(tmp_path, capsys):
    target = write(tmp_path)
    main(["check", str(target), "--diff"])
    assert "+        losses.append(loss.detach())" in capsys.readouterr().out
    assert target.read_text() == LEAKY


def test_fix_writes(tmp_path):
    target = write(tmp_path)
    main(["check", str(target), "--fix"])
    assert "loss.detach()" in target.read_text()


def test_rules_listing(capsys):
    assert main(["rules"]) == EXIT_OK
    out = capsys.readouterr().out
    for code in ("TG001", "TG002", "TG003", "TG004", "TG005"):
        assert code in out


def test_explain(capsys):
    assert main(["explain", "tg001"]) == EXIT_OK
    assert "autograd graph" in capsys.readouterr().out


def test_explain_unknown_rule(capsys):
    assert main(["explain", "TG999"]) == EXIT_ERROR


def test_unknown_code_warns_but_still_runs(tmp_path, capsys):
    assert main(["check", str(write(tmp_path)), "--ignore", "TG404"]) == EXIT_FINDINGS
    assert "unknown rule code TG404" in capsys.readouterr().err


# ------------------------------------------------------------ estimate / gpus

FINETUNE = textwrap.dedent(
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loader = DataLoader(ds, batch_size=4, num_workers=8, pin_memory=True)

    def train(device):
        for batch in loader:
            tokens = tokenizer(batch, max_length=2048)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**tokens).loss
            loss.backward()
            optimizer.step()
    """
).lstrip("\n")


def test_estimate_from_a_script(tmp_path, capsys):
    target = write(tmp_path, "finetune.py", FINETUNE)
    code = main(["estimate", str(target), "--gpu", "rtx4090", "-f", "json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_FINDINGS          # projected OOM must fail CI
    assert payload["model"]["name"] == "llama-2-7b"
    assert payload["config"]["batch_size"] == 4
    assert payload["config"]["seq_len"] == 2048
    assert payload["band"] in ("LIKELY_OOM", "CERTAIN_OOM")
    assert payload["remediations"]


def test_estimate_fits_returns_zero(capsys):
    code = main(["estimate", "--model", "distilbert-base-uncased", "--gpu", "a100-80gb",
                 "--batch-size", "8", "--seq-len", "128", "-f", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["band"] == "FITS"


def test_estimate_overrides_win_over_the_script(tmp_path, capsys):
    target = write(tmp_path, "finetune.py", FINETUNE)
    main(["estimate", str(target), "--gpu", "a100-80gb", "--batch-size", "1",
          "--seq-len", "512", "-f", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["batch_size"] == 1
    assert payload["config"]["seq_len"] == 512
    assert payload["config"]["sources"]["batch_size"] == "command line"


def test_estimate_with_explicit_params(capsys):
    main(["estimate", "--params", "7B", "--gpu", "a100-80gb", "-f", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"]["params"] == 7_000_000_000


def test_estimate_with_custom_gpu_memory(capsys):
    main(["estimate", "--params", "1B", "--gpu-memory", "48GiB", "-f", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpu"]["usable_bytes"] < 48 * 1024 ** 3


def test_estimate_unknown_model_is_reported_not_guessed(capsys):
    code = main(["estimate", "--model", "not-a-real-model", "--gpu", "a100-80gb",
                 "-f", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK                # unknown must not fail a build
    assert payload["resolved"] is False
    assert payload["band"] == "UNKNOWN"
    assert payload["total_bytes"] == 0


def test_estimate_rejects_unknown_gpu(capsys):
    assert main(["estimate", "--model", "gpt2", "--gpu", "rtx9090"]) == EXIT_ERROR


def test_estimate_needs_something_to_work_with():
    assert main(["estimate"]) == EXIT_ERROR


def test_gpus_listing(capsys):
    assert main(["gpus", "--instances"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "a100-80gb" in out
    assert "p4de.24xlarge" in out
