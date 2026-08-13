"""Rule-by-rule behaviour: what must fire, and what must stay quiet."""

from conftest import analyze, codes


# --------------------------------------------------------------------- TG001


def test_tg001_flags_appending_attached_loss():
    assert "TG001" in codes(
        """
        def train(model, loader, criterion, optimizer):
            losses = []
            for batch, y in loader:
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss)
        """
    )


def test_tg001_flags_augmented_accumulation():
    diagnostics = analyze(
        """
        def train(model, loader, criterion, optimizer):
            total = 0.0
            for batch, y in loader:
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                loss.backward()
                total += loss
        """
    )
    assert [d.code for d in diagnostics] == ["TG001"]
    assert "accumulates a graph-attached tensor" in diagnostics[0].message


def test_tg001_flags_dict_and_self_containers():
    assert "TG001" in codes(
        """
        def train(model, loader, criterion, optimizer, cache):
            for i, (batch, y) in enumerate(loader):
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                loss.backward()
                cache[i] = loss
        """
    )


def test_tg001_flags_self_attribute_container_without_a_loop():
    assert "TG001" in codes(
        """
        class Trainer:
            def training_step(self, batch, y):
                loss = self.criterion(self.model(batch), y)
                self.outputs.append(loss)
        """
    )


def test_tg001_quiet_when_detached_or_scalarised():
    for stored in ("loss.item()", "loss.detach()", "float(loss)", "loss.detach().cpu()"):
        assert codes(
            f"""
            def train(model, loader, criterion, optimizer):
                losses = []
                for batch, y in loader:
                    loss = criterion(model(batch), y)
                    optimizer.zero_grad()
                    loss.backward()
                    losses.append({stored})
            """
        ) == [], stored


def test_tg001_quiet_for_container_rebuilt_each_iteration():
    assert codes(
        """
        def train(model, loader, criterion, optimizer):
            for batch, y in loader:
                losses = []
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss)
        """
    ) == []


def test_tg001_quiet_inside_no_grad():
    assert codes(
        """
        import torch

        def collect(model, loader, criterion):
            outputs = []
            with torch.no_grad():
                for batch, y in loader:
                    outputs.append(model(batch))
        """
    ) == []


# --------------------------------------------------------------------- TG002


def test_tg002_flags_eval_function_without_no_grad():
    diagnostics = analyze(
        """
        import torch

        def validate(model, loader):
            for x, y in loader:
                out = model(x)
        """
    )
    assert [d.code for d in diagnostics] == ["TG002"]
    assert diagnostics[0].fixable


def test_tg002_offers_no_fix_when_torch_is_not_imported():
    """We cannot write ``@torch.no_grad()`` into a file that never imported torch."""
    diagnostics = analyze(
        """
        def validate(model, loader):
            for x, y in loader:
                out = model(x)
        """
    )
    assert [d.code for d in diagnostics] == ["TG002"]
    assert not diagnostics[0].fixable


def test_tg002_flags_inline_validation_loop_inside_training():
    diagnostics = analyze(
        """
        def train(model, loader, val_loader, criterion, optimizer):
            for x, y in loader:
                optimizer.zero_grad()
                criterion(model(x), y).backward()
            for x, y in val_loader:
                out = model(x)
        """
    )
    assert [d.code for d in diagnostics] == ["TG002"]
    # Never offer to decorate a training routine with @torch.no_grad().
    assert not diagnostics[0].fixable


def test_tg002_quiet_when_guarded():
    for guard in ("@torch.no_grad()", "@torch.inference_mode()", "@torch.no_grad"):
        assert codes(
            f"""
            import torch

            {guard}
            def validate(model, loader):
                for x, y in loader:
                    out = model(x)
            """
        ) == [], guard


def test_tg002_quiet_with_context_manager():
    assert codes(
        """
        import torch

        def validate(model, loader):
            with torch.no_grad():
                for x, y in loader:
                    out = model(x)
        """
    ) == []


def test_tg002_quiet_for_pytest_functions():
    assert codes(
        """
        def test_forward_shape(model, x):
            out = model(x)
            assert out.shape == (2, 10)
        """
    ) == []


def test_tg002_quiet_for_lightning_hooks():
    assert codes(
        """
        import pytorch_lightning as pl

        class Lit(pl.LightningModule):
            def validation_step(self, batch, idx):
                x, y = batch
                return self.model(x)
        """
    ) == []


# --------------------------------------------------------------------- TG003


def test_tg003_flags_backward_without_zero_grad():
    assert "TG003" in codes(
        """
        def train(model, loader, criterion, optimizer):
            for batch, y in loader:
                loss = criterion(model(batch), y)
                loss.backward()
                optimizer.step()
        """
    )


def test_tg003_quiet_when_zero_grad_present():
    assert codes(
        """
        def train(model, loader, criterion, optimizer):
            for batch, y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch), y)
                loss.backward()
                optimizer.step()
        """
    ) == []


def test_tg003_quiet_for_gradient_accumulation():
    """``zero_grad`` guarded by an ``if`` is deliberate accumulation, not a bug."""
    assert codes(
        """
        def train(model, loader, criterion, optimizer, accum=4):
            for i, (batch, y) in enumerate(loader):
                loss = criterion(model(batch), y)
                loss.backward()
                if (i + 1) % accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        """
    ) == []


def test_tg003_quiet_outside_a_loop():
    assert codes(
        """
        def one_step(model, batch, y, criterion, optimizer):
            loss = criterion(model(batch), y)
            loss.backward()
            optimizer.step()
        """
    ) == []


def test_tg003_quiet_in_lightning_module():
    assert codes(
        """
        import lightning as L

        class Lit(L.LightningModule):
            def training_step(self, batch, idx):
                x, y = batch
                for _ in range(2):
                    loss = self.criterion(self.model(x), y)
                    loss.backward()
                return loss
        """
    ) == []


# --------------------------------------------------------------------- TG004

_CUDA_PREAMBLE = """
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda")
"""


def test_tg004_flags_missing_workers_and_pin_memory():
    diagnostics = analyze(_CUDA_PREAMBLE + "loader = DataLoader(ds, batch_size=32)\n")
    assert [d.code for d in diagnostics] == ["TG004", "TG004"]
    assert any("num_workers" in d.message for d in diagnostics)
    assert any("pin_memory" in d.message for d in diagnostics)


def test_tg004_flags_explicit_zero_workers():
    diagnostics = analyze(
        _CUDA_PREAMBLE + "loader = DataLoader(ds, num_workers=0, pin_memory=True)\n"
    )
    assert [d.code for d in diagnostics] == ["TG004"]
    assert "num_workers=0" in diagnostics[0].message


def test_tg004_quiet_when_configured():
    assert codes(
        _CUDA_PREAMBLE + "loader = DataLoader(ds, num_workers=8, pin_memory=True)\n"
    ) == []


def test_tg004_quiet_without_a_gpu_target():
    assert codes(
        """
        from torch.utils.data import DataLoader

        loader = DataLoader(ds, batch_size=32)
        """
    ) == []


# --------------------------------------------------------------------- TG005


def test_tg005_flags_softmax_wrapped_in_cross_entropy():
    diagnostics = analyze(
        """
        import torch.nn.functional as F

        def step(model, x, y):
            logits = model(x)
            return F.cross_entropy(F.softmax(logits, dim=1), y)
        """
    )
    assert [d.code for d in diagnostics] == ["TG005"]
    assert diagnostics[0].fixable


def test_tg005_flags_criterion_variable():
    assert "TG005" in codes(
        """
        import torch
        import torch.nn as nn

        criterion = nn.CrossEntropyLoss()

        def step(model, x, y):
            return criterion(torch.softmax(model(x), dim=1), y)
        """
    )


def test_tg005_flags_indirect_softmax_variable():
    diagnostics = analyze(
        """
        import torch.nn as nn
        import torch.nn.functional as F

        criterion = nn.CrossEntropyLoss()

        def step(model, x, y):
            probs = F.softmax(model(x), dim=1)
            return criterion(probs, y)
        """
    )
    assert [d.code for d in diagnostics] == ["TG005"]
    assert not diagnostics[0].fixable


def test_tg005_flags_softmax_layer_in_model():
    assert "TG005" in codes(
        """
        import torch.nn as nn

        criterion = nn.CrossEntropyLoss()
        head = nn.Sequential(nn.Linear(8, 4), nn.Softmax(dim=1))
        """
    )


def test_tg005_quiet_on_raw_logits():
    assert codes(
        """
        import torch.nn.functional as F

        def step(model, x, y):
            return F.cross_entropy(model(x), y)
        """
    ) == []


def test_tg005_quiet_for_nll_with_log_softmax():
    """``NLLLoss`` genuinely wants ``log_softmax`` — this pairing is correct."""
    assert codes(
        """
        import torch.nn.functional as F

        def step(model, x, y):
            return F.nll_loss(F.log_softmax(model(x), dim=1), y)
        """
    ) == []


def test_tg005_flags_softmax_passed_to_nll():
    assert "TG005" in codes(
        """
        import torch.nn.functional as F

        def step(model, x, y):
            return F.nll_loss(F.softmax(model(x), dim=1), y)
        """
    )


# ------------------------------------------------------------------ examples


def test_good_example_is_clean():
    from pathlib import Path

    from torch_preflight.engine import check_source

    path = Path(__file__).parent.parent / "examples" / "good_train.py"
    diagnostics, _ = check_source(str(path), path.read_text())
    assert diagnostics == [], [d.message for d in diagnostics]


def test_bad_example_triggers_every_rule():
    from pathlib import Path

    from torch_preflight.engine import check_source

    path = Path(__file__).parent.parent / "examples" / "bad_train.py"
    diagnostics, _ = check_source(str(path), path.read_text())
    assert {d.code for d in diagnostics} == {"TG001", "TG002", "TG003", "TG004", "TG005"}


def test_tg001_quiet_for_non_differentiable_outputs():
    """``argmax`` returns indices with no graph — storing them is safe."""
    for stored in (
        "logits.argmax(-1)",
        "logits.argmax(dim=1)",
        "(logits > 0).long()",
        "logits.topk(5).indices",
        "torch.argmax(logits, dim=1)",
    ):
        assert codes(
            f"""
            import torch

            def evaluate(model, loader):
                preds = []
                with torch.no_grad():
                    for x, y in loader:
                        logits = model(x)
                        preds.append({stored})
                return preds
            """
        ) == [], stored


def test_tg001_still_flags_differentiable_reductions():
    """``.sum()``/``.mean()`` keep the graph, so they must still be caught."""
    for stored in ("loss.sum()", "loss.mean()", "loss.float()", "loss * 2"):
        assert "TG001" in codes(
            f"""
            def train(model, loader, criterion, optimizer):
                losses = []
                for batch, y in loader:
                    loss = criterion(model(batch), y)
                    optimizer.zero_grad()
                    loss.backward()
                    losses.append({stored})
            """
        ), stored


def test_tg003_quiet_without_an_optimizer_step():
    """Raw autograd on a tensor recreated each iteration accumulates nothing.

    Found on torch's own test suite: a fresh leaf per iteration means `.grad` starts at
    None every time, so there is no stale gradient to clear.
    """
    assert codes(
        """
        import torch

        def bench(model):
            for _ in range(10):
                x = torch.rand([1000, 1000], requires_grad=True)
                loss = (x * 2.0).sum()
                loss.backward()
        """
    ) == []


def test_tg003_ignores_scheduler_step():
    """`scheduler.step()` advances the LR; it applies no gradients."""
    assert "TG003" not in codes(
        """
        def train(model, loader, criterion, scheduler):
            for batch, y in loader:
                loss = criterion(model(batch), y)
                loss.backward()
                scheduler.step()
        """
    )


def test_tg003_still_fires_with_an_optimizer_step():
    assert "TG003" in codes(
        """
        def train(model, loader, criterion, optimizer):
            for batch, y in loader:
                loss = criterion(model(batch), y)
                loss.backward()
                optimizer.step()
        """
    )


# ------------------------------- regressions found on torch's own source (triage pass)


def test_tg002_eval_loader_needs_to_look_like_a_loader():
    """`models_to_test` is not a validation dataloader.

    In a test suite half the identifiers contain "test"; matching on that alone made
    TG002 fire on a training loop inside torch's distributed tests.
    """
    assert codes(
        """
        def _test_ddp_parity(models_to_test, inp, optimizer):
            for model in models_to_test:
                for _ in range(6):
                    optimizer.zero_grad()
                    out = model(inp)
                    out.sum().backward()
                    optimizer.step()
        """
    ) == []


def test_tg002_still_fires_on_a_real_validation_loader():
    assert "TG002" in codes(
        """
        def train(model, train_loader, val_loader, criterion, optimizer):
            for x, y in train_loader:
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()
            for x, y in val_loader:
                out = model(x)
        """
    )


def test_a_bare_prepare_call_is_not_a_model_wrapper():
    """torch's quantization utilities call their own function `prepare`."""
    assert codes(
        """
        def check(model, inputs, qconfig):
            model.eval()
            prepared = prepare(model, qconfig, example_inputs=inputs)
            prepared(*inputs)
        """
    ) == []


def test_accelerator_prepare_is_still_a_model_wrapper():
    assert "TG002" in codes(
        """
        def evaluate(accelerator, raw_model, loader):
            model = accelerator.prepare(raw_model)
            for batch in loader:
                out = model(batch)
        """
    )


def test_inner_scope_binding_shadows_an_outer_grad_name():
    """A nested helper with its own `loss` must not inherit grad-ness from outside."""
    assert codes(
        """
        def test_something(model, loader, criterion, optimizer):
            for batch, y in loader:
                loss = criterion(model(batch), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            def get_loss(model_output):
                loss = 0.0
                for value in model_output.values():
                    loss += get_loss(value)
                return loss
        """
    ) == []


def test_autograd_grad_results_are_detached():
    """`torch.autograd.grad` returns detached tensors unless create_graph=True."""
    assert codes(
        """
        import torch

        def warmup_backward(f, *args):
            results = []
            for _ in range(3):
                r = torch.autograd.grad(f, *args)
                results.append(r)
            return results
        """
    ) == []


def test_autograd_grad_with_create_graph_still_retains():
    assert "TG001" in codes(
        """
        import torch

        def higher_order(f, *args):
            results = []
            for _ in range(3):
                r = torch.autograd.grad(f, *args, create_graph=True)
                results.append(r)
            return results
        """
    )


def test_model_names_do_not_leak_across_functions():
    """A model-ish binding in one function must not rename an unrelated local elsewhere.

    From torch's quantization utilities: `prepared = DistributedDataParallel(...)` in one
    helper made `prepared = prepare(...)` a thousand lines away look like a model.
    """
    assert codes(
        """
        from torch.nn.parallel import DistributedDataParallel

        def setup(rank, prepared):
            prepared = DistributedDataParallel(prepared, device_ids=[rank])
            return prepared

        def check_graph_op(model, inputs, qconfig):
            model.eval()
            prepared = prepare(model, qconfig, example_inputs=inputs)
            prepared(*inputs)
        """
    ) == []


def test_a_model_in_an_enclosing_scope_is_still_visible():
    """Scope-awareness must not break the ordinary nested-use case."""
    assert "TG002" in codes(
        """
        import torch
        from torch.nn.parallel import DistributedDataParallel

        def run(base, val_loader):
            wrapped = DistributedDataParallel(base)

            def evaluate():
                for batch in val_loader:
                    out = wrapped(batch)

            evaluate()
        """
    )
