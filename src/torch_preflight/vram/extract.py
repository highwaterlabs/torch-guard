"""Extract a :class:`RunConfig` from a training script.

Reuses the existing CST analysis layer — the same traversal the lint rules run on. Nothing
is imported or executed; every value here comes from reading the source.

Each field records where it came from, so the report can say ``batch_size=64 (train.py:27)``
and the user can see what we assumed and why.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import libcst as cst
from libcst.metadata import PositionProvider

from ..analysis.context import FileContext, build_context
from ..analysis.helpers import dotted_name, final_attr, keyword_arg, positional_args
from ..analysis.scope import ScopeTrackingVisitor, target_names
from .types import OptimizerKind, PrecisionMode, RunConfig, Sharding

_OPTIMIZERS = {
    "sgd": OptimizerKind.SGD,
    "adam": OptimizerKind.ADAM,
    "adamw": OptimizerKind.ADAMW,
    "nadam": OptimizerKind.ADAMW,
    "radam": OptimizerKind.ADAMW,
    "adamw8bit": OptimizerKind.ADAM_8BIT,
    "adam8bit": OptimizerKind.ADAM_8BIT,
    "pagedadamw8bit": OptimizerKind.ADAM_8BIT,
    "pagedadamw": OptimizerKind.ADAMW,
    "adafactor": OptimizerKind.ADAFACTOR,
    "lion": OptimizerKind.LION,
    "lion8bit": OptimizerKind.LION_8BIT,
    "rmsprop": OptimizerKind.RMSPROP,
    "adagrad": OptimizerKind.ADAGRAD,
}

_SEQ_LEN_KEYS = (
    "max_length", "max_seq_length", "max_seq_len", "block_size", "seq_len",
    "sequence_length", "context_length", "max_position_embeddings", "n_ctx",
)

_IMAGE_SIZE_KEYS = ("image_size", "img_size", "resolution", "input_size")

_BATCH_KEYS = (
    "batch_size", "per_device_train_batch_size", "train_batch_size",
    "micro_batch_size", "per_gpu_train_batch_size",
)

_ACCUM_KEYS = ("gradient_accumulation_steps", "accumulate_grad_batches", "accum_steps",
               "accumulation_steps", "grad_accum", "gradient_accumulation")

_EVAL_HINTS = ("val", "valid", "test", "eval", "dev")


@dataclass
class ExtractedConfig:
    config: RunConfig
    model_ref: Optional[str] = None
    model_ref_line: Optional[int] = None
    #: Unresolvable model construction sites, for the "could not resolve" message.
    unresolved_models: List[Tuple[str, int]] = field(default_factory=list)


class _Extractor(ScopeTrackingVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.constants: Dict[str, int] = {}
        #: Dict literals by name, so a DeepSpeed config passed by variable can be read.
        self.dict_literals: Dict[str, cst.Dict] = {}
        #: True when a DeepSpeed stage had to be assumed rather than read.
        self.deepspeed_stage_guessed = False
        self.sources: Dict[str, str] = {}

        self.batch_candidates: List[Tuple[int, bool, int]] = []  # (value, is_eval, line)
        self.seq_len: Optional[int] = None
        self.image_size: Optional[int] = None
        self.optimizer: Optional[OptimizerKind] = None
        self.accumulation: int = 1
        self.world_size: int = 1
        self.sharding: Sharding = Sharding.NONE

        self.saw_autocast = False
        self.saw_grad_scaler = False
        self.saw_pure_cast = False
        self.saw_fp16_master = False
        self.gradient_checkpointing = False
        self.flash_attention = False
        self.saw_backward = False

        self.model_ref: Optional[str] = None
        self.model_ref_line: Optional[int] = None
        self.unresolved: List[Tuple[str, int]] = []

    # ------------------------------------------------------------------ utilities

    def _line(self, node: cst.CSTNode) -> int:
        return self.get_metadata(PositionProvider, node).start.line

    def _record(self, field_name: str, node: cst.CSTNode) -> None:
        self.sources[field_name] = f"{self.path}:{self._line(node)}"

    def _int_of(self, node: cst.BaseExpression) -> Optional[int]:
        """Fold an integer literal, or a name bound to one at module level."""
        if isinstance(node, cst.Integer):
            try:
                return int(node.value)
            except ValueError:
                return None
        if isinstance(node, cst.Name):
            return self.constants.get(node.value)
        if isinstance(node, cst.Attribute):
            return self.constants.get(dotted_name(node) or "")
        if isinstance(node, cst.UnaryOperation):
            return None
        return None

    # --------------------------------------------------------- module-level constants

    def visit_Assign(self, node: cst.Assign) -> bool:
        value = self._int_of(node.value)
        if value is not None:
            for target in node.targets:
                for name in target_names(target.target):
                    self.constants[name] = value

        # DeepSpeed configs are usually a dict literal assigned to a name and handed to
        # ``deepspeed.initialize(config=...)``, so keep the node to read the stage from.
        if isinstance(node.value, cst.Dict):
            for target in node.targets:
                for name in target_names(target.target):
                    self.dict_literals[name] = node.value

        # ``model.half()`` / ``model.bfloat16()`` casts the parameters themselves.
        if isinstance(node.value, cst.Call):
            leaf = final_attr(node.value.func)
            if leaf in ("half", "bfloat16") and self.prov_is_model(node.value):
                self.saw_pure_cast = True
                self._record("precision", node)
        return True

    def prov_is_model(self, call: cst.Call) -> bool:
        func = call.func
        if not isinstance(func, cst.Attribute):
            return False
        base = dotted_name(func.value) or ""
        return base.rsplit(".", 1)[-1] in {
            "model", "net", "network", "module", "backbone", "encoder", "decoder",
        }

    # ------------------------------------------------------------------------ calls

    def visit_Call(self, node: cst.Call) -> bool:
        leaf = final_attr(node.func) or ""
        dotted = dotted_name(node.func) or ""
        lowered = leaf.lower()

        self._check_batch_size(node, leaf)
        self._check_keywords(node)
        self._check_optimizer(node, lowered, dotted)
        self._check_precision(node, leaf, dotted)
        self._check_sharding(node, leaf)
        self._check_model_ref(node, leaf)
        self._check_transforms(node, leaf)

        if leaf == "backward":
            self.saw_backward = True
        if leaf in ("gradient_checkpointing_enable", "checkpoint") or "checkpoint" in lowered:
            if leaf != "load_checkpoint":
                self.gradient_checkpointing = True
                self._record("gradient_checkpointing", node)
        if leaf == "scaled_dot_product_attention":
            self.flash_attention = True
            self._record("flash_attention", node)
        return True

    def _check_batch_size(self, node: cst.Call, leaf: str) -> None:
        for key in _BATCH_KEYS:
            arg = keyword_arg(node, key)
            if arg is None:
                continue
            value = self._int_of(arg.value)
            if value is None:
                continue
            is_eval = self._looks_like_eval_context(node)
            self.batch_candidates.append((value, is_eval, self._line(node)))

    def _looks_like_eval_context(self, node: cst.Call) -> bool:
        """A DataLoader assigned to ``val_loader`` should not drive the estimate."""
        for arg in node.args:
            name = dotted_name(arg.value) or ""
            if any(hint in name.lower() for hint in _EVAL_HINTS):
                return True
        return False

    def _check_keywords(self, node: cst.Call) -> None:
        for key in _SEQ_LEN_KEYS:
            arg = keyword_arg(node, key)
            if arg is not None and self.seq_len is None:
                value = self._int_of(arg.value)
                if value:
                    self.seq_len = value
                    self._record("seq_len", node)

        for key in _IMAGE_SIZE_KEYS:
            arg = keyword_arg(node, key)
            if arg is not None and self.image_size is None:
                value = self._int_of(arg.value)
                if value:
                    self.image_size = value
                    self._record("image_size", node)

        for key in _ACCUM_KEYS:
            arg = keyword_arg(node, key)
            if arg is not None:
                value = self._int_of(arg.value)
                if value and value > 1:
                    self.accumulation = value
                    self._record("accumulation_steps", node)

        arg = keyword_arg(node, "world_size")
        if arg is not None:
            value = self._int_of(arg.value)
            if value:
                self.world_size = value
                self._record("world_size", node)

        for key in ("gradient_checkpointing", "use_gradient_checkpointing"):
            arg = keyword_arg(node, key)
            if arg is not None and isinstance(arg.value, cst.Name) and arg.value.value == "True":
                self.gradient_checkpointing = True
                self._record("gradient_checkpointing", node)

        for key in ("use_flash_attention_2", "use_flash_attn"):
            arg = keyword_arg(node, key)
            if arg is not None and isinstance(arg.value, cst.Name) and arg.value.value == "True":
                self.flash_attention = True
                self._record("flash_attention", node)

        arg = keyword_arg(node, "attn_implementation")
        if arg is not None and isinstance(arg.value, cst.SimpleString):
            if "flash" in arg.value.value.lower() or "sdpa" in arg.value.value.lower():
                self.flash_attention = True
                self._record("flash_attention", node)

    def _check_optimizer(self, node: cst.Call, lowered: str, dotted: str) -> None:
        kind = _OPTIMIZERS.get(lowered.replace("_", ""))
        if kind is None:
            return
        # Only treat it as the optimizer if it is plausibly from an optim namespace.
        if not (
            "optim" in dotted.lower()
            or dotted.lower().startswith(("bnb.", "bitsandbytes."))
            or dotted == lowered
            or "." not in dotted
        ):
            return

        if kind is OptimizerKind.SGD:
            momentum = keyword_arg(node, "momentum")
            if momentum is not None and not _is_zero(momentum.value):
                kind = OptimizerKind.SGD_MOMENTUM

        self.optimizer = kind
        self._record("optimizer", node)

    def _check_precision(self, node: cst.Call, leaf: str, dotted: str) -> None:
        if leaf == "autocast":
            self.saw_autocast = True
            self._record("precision", node)
        elif leaf == "GradScaler":
            self.saw_grad_scaler = True
            self._record("precision", node)
        elif leaf in ("half", "bfloat16") and self.prov_is_model(node):
            self.saw_pure_cast = True
            self._record("precision", node)

        for key in ("fp16", "bf16"):
            arg = keyword_arg(node, key)
            if arg is not None and isinstance(arg.value, cst.Name) and arg.value.value == "True":
                self.saw_autocast = True
                self._record("precision", node)

        arg = keyword_arg(node, "precision")
        if arg is not None and isinstance(arg.value, cst.SimpleString):
            text = arg.value.value.lower()
            if "mixed" in text or "16" in text:
                self.saw_autocast = True
                self._record("precision", node)
            if "true" in text and "16" in text:
                self.saw_pure_cast = True

    def _check_sharding(self, node: cst.Call, leaf: str) -> None:
        if leaf in ("DistributedDataParallel", "DDP"):
            self.sharding = Sharding.DDP
            self._record("sharding", node)
        elif leaf in ("FullyShardedDataParallel", "FSDP"):
            self.sharding = Sharding.ZERO3
            self._record("sharding", node)
        elif leaf == "initialize" and "deepspeed" in (dotted_name(node.func) or ""):
            self.saw_fp16_master = True
            self._read_deepspeed_stage(node)
            self._record("sharding", node)
        elif leaf == "TrainingArguments" and keyword_arg(node, "deepspeed") is not None:
            # HF takes the same config, by path or dict, under its own keyword.
            self.saw_fp16_master = True
            self._read_deepspeed_stage(node, keyword="deepspeed")
            self._record("sharding", node)


    # ------------------------------------------------------------- DeepSpeed config

    #: DeepSpeed ZeRO stage -> our sharding model. Stage 0 is plain data parallelism.
    _ZERO_STAGES = {
        0: Sharding.DDP,
        1: Sharding.ZERO1,
        2: Sharding.ZERO2,
        3: Sharding.ZERO3,
    }

    def _read_deepspeed_stage(self, node: cst.Call, keyword: str = "") -> None:
        """Resolve the ZeRO stage instead of assuming it.

        The stage changes the estimate substantially -- stage 3 shards parameters as well
        as optimizer state, so it is the difference between a 70B model fitting and not.
        We used to assume stage 2 because "the stage is in a JSON config we cannot read",
        but in practice it is readable: it is either a dict literal in the same file or a
        path to a JSON file sitting next to it. Neither requires executing anything.

        Falls back to the old stage-2 assumption when the config cannot be resolved, and
        records that it was a guess.
        """
        config = (
            keyword_arg(node, keyword) if keyword
            else keyword_arg(node, "config") or keyword_arg(node, "config_params")
        )
        stage = self._zero_stage_from(config.value) if config is not None else None

        if stage is None:
            self.sharding = Sharding.ZERO2
            self.deepspeed_stage_guessed = True
            return
        self.sharding = self._ZERO_STAGES.get(stage, Sharding.ZERO2)
        self.deepspeed_stage_guessed = False

    def _zero_stage_from(self, value: cst.BaseExpression) -> Optional[int]:
        if isinstance(value, cst.Name):
            literal = self.dict_literals.get(value.value)
            return self._stage_in_dict(literal) if literal is not None else None
        if isinstance(value, cst.Dict):
            return self._stage_in_dict(value)
        if isinstance(value, cst.SimpleString):
            return self._stage_in_json(_string_value(value))
        return None

    def _stage_in_dict(self, node: cst.Dict) -> Optional[int]:
        for element in node.elements:
            if not isinstance(element, cst.DictElement):
                continue
            key = element.key
            if not isinstance(key, cst.SimpleString):
                continue
            if _string_value(key) != "zero_optimization":
                continue
            if not isinstance(element.value, cst.Dict):
                continue
            for inner in element.value.elements:
                if not isinstance(inner, cst.DictElement):
                    continue
                if isinstance(inner.key, cst.SimpleString) and _string_value(inner.key) == "stage":
                    return self._int_of(inner.value)
        return None

    def _stage_in_json(self, relative: str) -> Optional[int]:
        """Read ``zero_optimization.stage`` from a JSON config beside the source file.

        Reading a JSON file is not executing code, which is the line that matters for
        ``check``. Any failure -- missing file, bad JSON, absolute path outside the tree --
        simply leaves the stage unresolved.
        """
        if not relative or os.path.isabs(relative):
            return None
        base = os.path.dirname(os.path.abspath(self.path))
        candidate = os.path.normpath(os.path.join(base, relative))
        if not candidate.startswith(base) or not os.path.isfile(candidate):
            return None
        try:
            with open(candidate, encoding="utf-8") as handle:
                data: Any = json.load(handle)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        zero = data.get("zero_optimization")
        if not isinstance(zero, dict):
            return None
        stage = zero.get("stage")
        return stage if isinstance(stage, int) else None

    def _check_model_ref(self, node: cst.Call, leaf: str) -> None:
        if leaf != "from_pretrained" or self.model_ref is not None:
            return
        args = positional_args(node)
        if args and isinstance(args[0].value, cst.SimpleString):
            self.model_ref = _string_value(args[0].value)
            self.model_ref_line = self._line(node)
        elif args:
            self.unresolved.append((_render(args[0].value), self._line(node)))

    def _check_transforms(self, node: cst.Call, leaf: str) -> None:
        if leaf not in ("Resize", "CenterCrop", "RandomResizedCrop"):
            return
        args = positional_args(node)
        if not args:
            return
        value = args[0].value
        if isinstance(value, (cst.Tuple, cst.List)) and value.elements:
            size = self._int_of(value.elements[0].value)
        else:
            size = self._int_of(value)
        if size and self.image_size is None:
            self.image_size = size
            self._record("image_size", node)

    # ---------------------------------------------------------------------- result

    def precision(self) -> PrecisionMode:
        if self.saw_fp16_master:
            return PrecisionMode.FP16_MASTER
        if self.saw_pure_cast:
            return PrecisionMode.PURE_BF16
        if self.saw_autocast or self.saw_grad_scaler:
            return PrecisionMode.AMP
        return PrecisionMode.FP32

    def batch_size(self) -> Tuple[int, Optional[int]]:
        """Prefer training loaders; among them the largest, since that drives the peak."""
        if not self.batch_candidates:
            return 1, None
        train = [c for c in self.batch_candidates if not c[1]] or self.batch_candidates
        value, _, line = max(train, key=lambda c: c[0])
        return value, line


def _is_zero(node: cst.BaseExpression) -> bool:
    if isinstance(node, cst.Integer):
        return node.value.strip() == "0"
    if isinstance(node, cst.Float):
        try:
            return float(node.value) == 0.0
        except ValueError:
            return False
    return False


def _string_value(node: cst.SimpleString) -> str:
    return node.value.strip("\"'")


def _render(node: cst.BaseExpression) -> str:
    return cst.Module(body=[]).code_for_node(node)


def extract(ctx: FileContext) -> ExtractedConfig:
    """Read a :class:`RunConfig` out of an already-parsed file."""
    extractor = _Extractor(ctx.path)
    ctx.wrapper.visit(extractor)

    batch, batch_line = extractor.batch_size()
    if batch_line is not None:
        extractor.sources["batch_size"] = f"{ctx.path}:{batch_line}"

    config = RunConfig(
        batch_size=batch,
        seq_len=extractor.seq_len,
        image_size=extractor.image_size,
        precision=extractor.precision(),
        optimizer=extractor.optimizer or OptimizerKind.ADAMW,
        gradient_checkpointing=extractor.gradient_checkpointing,
        flash_attention=extractor.flash_attention,
        accumulation_steps=extractor.accumulation,
        world_size=extractor.world_size,
        sharding=extractor.sharding,
        inference_only=not extractor.saw_backward,
        sources=extractor.sources,
    )

    return ExtractedConfig(
        config=config,
        model_ref=extractor.model_ref,
        model_ref_line=extractor.model_ref_line,
        unresolved_models=extractor.unresolved,
    )


def extract_from_source(path: str, source: str) -> ExtractedConfig:
    return extract(build_context(path, source))
