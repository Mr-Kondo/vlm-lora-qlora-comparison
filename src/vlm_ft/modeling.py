"""Model loading, the LoRA/QLoRA method switch, and adapter construction.

The *only* intentional differences between the two fine-tuning methods live in
:data:`METHOD_SPECS` and the two ``if spec...`` branches below:

    lora  -> base weights loaded in the 16-bit compute dtype
    qlora -> the SAME base weights loaded 4-bit NF4, then prepared for k-bit
             training

Everything else -- the adapter configuration, the resolved target modules, the
dtype used for compute, the optimizer, the data -- is shared.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

LOGGER = logging.getLogger(__name__)

_DTYPE_ALIASES = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
}

#: Leaf module names that identify a sub-tower, used by ``lora.target_scope``.
_SCOPE_LEAF_NAMES = {
    "text_model": ("text_model", "language_model"),
    "vision_model": ("vision_model", "vision_tower"),
}

_QUANT_LINEAR_NAMES = {"Linear4bit", "Linear8bitLt", "LinearNF4", "LinearFP4"}


@dataclass(frozen=True)
class MethodSpec:
    """The complete set of behaviours that differ between the two methods."""

    name: str
    quantize_base: bool
    prepare_for_kbit: bool
    description: str


METHOD_SPECS = {
    "lora": MethodSpec(
        name="lora",
        quantize_base=False,
        prepare_for_kbit=False,
        description="base weights held in the 16-bit compute dtype; LoRA adapters trained",
    ),
    "qlora": MethodSpec(
        name="qlora",
        quantize_base=True,
        prepare_for_kbit=True,
        description="base weights quantized to 4-bit NF4; identical LoRA adapters trained",
    ),
}


def get_method_spec(method: str) -> MethodSpec:
    try:
        return METHOD_SPECS[method]
    except KeyError:
        raise ValueError(
            f"unknown method {method!r}; expected one of {sorted(METHOD_SPECS)}"
        ) from None


def validate_method_against_config(method: str, cfg) -> MethodSpec:
    """Fail loudly when the CLI method and the config file disagree.

    A silent mismatch (``--method lora`` with a 4-bit config) would quietly
    destroy the controlled comparison, so it is an error rather than a warning.
    """
    spec = get_method_spec(method)
    declared = cfg.get("method")
    if declared is not None and declared != method:
        raise ValueError(
            f"config declares method={declared!r} but --method={method!r} was requested; "
            f"pass the matching config (configs/{method}.yaml)"
        )
    load_in_4bit = bool(cfg.quantization.get("load_in_4bit", False))
    if load_in_4bit != spec.quantize_base:
        raise ValueError(
            f"method {method!r} requires quantization.load_in_4bit={spec.quantize_base} "
            f"but the config sets {load_in_4bit}"
        )
    return spec


# --------------------------------------------------------------------------
# dtype / paths / revisions
# --------------------------------------------------------------------------
def resolve_compute_dtype(name: str | None):
    """Map ``auto``/alias to a concrete torch dtype.

    ``auto`` picks bfloat16 when the GPU supports it (Ampere and newer), else
    float16 on CUDA, else float32 on CPU. Whatever is chosen is used for LoRA's
    base weights *and* as QLoRA's ``bnb_4bit_compute_dtype``, so both methods
    compute in the same precision.
    """
    import torch

    if name in (None, "auto"):
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    key = _DTYPE_ALIASES.get(str(name).lower())
    if key is None:
        raise ValueError(f"unsupported dtype {name!r}; expected one of {sorted(_DTYPE_ALIASES)}")
    return getattr(torch, key)


def dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def model_source(cfg) -> str:
    """Local snapshot directory when one was downloaded, else the Hub id."""
    local_dir = cfg.model.get("local_dir")
    return local_dir if local_dir else cfg.model.id


def resolve_revision(cfg) -> dict:
    """Resolve ``model.revision`` to a concrete commit SHA for the record."""
    requested = cfg.model.get("revision") or "main"
    if cfg.model.get("local_dir"):
        return {"requested": requested, "resolved_sha": None, "source": "local_dir"}
    try:
        from huggingface_hub import model_info

        info = model_info(cfg.model.id, revision=requested)
        return {"requested": requested, "resolved_sha": info.sha, "source": "hub"}
    except Exception as exc:  # offline or rate-limited: record why, do not guess
        LOGGER.warning("could not resolve model revision: %s", exc)
        return {"requested": requested, "resolved_sha": None, "source": f"unresolved: {exc}"}


def _auto_model_class():
    import transformers

    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError(
        "transformers exposes neither AutoModelForImageTextToText nor "
        "AutoModelForVision2Seq; install transformers>=4.45"
    )


def _dtype_kwarg() -> str:
    """``dtype`` on transformers>=5, ``torch_dtype`` before that."""
    import transformers

    major = int(str(transformers.__version__).split(".")[0])
    return "dtype" if major >= 5 else "torch_dtype"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def build_quantization_config(cfg, spec: MethodSpec, compute_dtype):
    """BitsAndBytesConfig for QLoRA; ``None`` for LoRA."""
    if not spec.quantize_base:
        return None
    from transformers import BitsAndBytesConfig

    quant_cfg = cfg.quantization
    bnb_dtype = quant_cfg.get("bnb_4bit_compute_dtype", "auto")
    bnb_compute_dtype = compute_dtype if bnb_dtype in (None, "auto") else resolve_compute_dtype(bnb_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=bnb_compute_dtype,
    )


def load_processor(cfg):
    """Load the processor and apply the shared image-resolution settings."""
    from transformers import AutoProcessor

    from .data import configure_processor

    processor = AutoProcessor.from_pretrained(
        model_source(cfg), revision=cfg.model.get("revision") or "main"
    )
    return configure_processor(processor, cfg)


def load_base_model(cfg, spec: MethodSpec, compute_dtype, quantization_config=None):
    """Load the base VLM. Identical call for both methods except quantization."""
    import torch

    model_cls = _auto_model_class()
    kwargs = {
        _dtype_kwarg(): compute_dtype,
        "attn_implementation": cfg.model.get("attn_implementation") or "sdpa",
        "revision": cfg.model.get("revision") or "main",
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": torch.cuda.current_device()}

    LOGGER.info(
        "loading base model %s (method=%s, dtype=%s, 4bit=%s)",
        model_source(cfg),
        spec.name,
        dtype_name(compute_dtype),
        quantization_config is not None,
    )
    return model_cls.from_pretrained(model_source(cfg), **kwargs)


# --------------------------------------------------------------------------
# LoRA target resolution
# --------------------------------------------------------------------------
def _is_linear(module) -> bool:
    import torch.nn as nn

    return isinstance(module, nn.Linear) or type(module).__name__ in _QUANT_LINEAR_NAMES


def resolve_scope_prefix(model, scope: str) -> str:
    """Dotted prefix of the requested sub-tower, or ``""`` for the whole model."""
    if scope in (None, "", "all"):
        return ""
    leaf_names = _SCOPE_LEAF_NAMES.get(scope, (scope,))
    candidates = [
        name for name, _ in model.named_modules() if name and name.split(".")[-1] in leaf_names
    ]
    if not candidates:
        available = sorted(
            {n.split(".")[-1] for n, _ in model.named_modules() if n and "." not in n.split(".")[-1]}
        )
        raise ValueError(
            f"lora.target_scope={scope!r} matched no submodule. "
            f"Use 'all' or one of these leaf names: {available[:40]}"
        )
    # outermost match wins
    return min(candidates, key=lambda n: (n.count("."), len(n))) + "."


def resolve_target_modules(model, suffixes: Iterable[str], scope: str) -> list[str]:
    """Fully-qualified names of the Linear modules LoRA will wrap.

    Resolving against the *instantiated* model matters here: SmolVLM's vision
    tower also contains ``q_proj``/``k_proj``/``v_proj``, so a bare suffix list
    handed straight to PEFT would silently adapt the vision encoder too.
    Returning explicit names makes the targeted set auditable and lets the LoRA
    and QLoRA runs be checked for equality after the fact.
    """
    suffixes = set(suffixes)
    prefix = resolve_scope_prefix(model, scope)
    names = sorted(
        name
        for name, module in model.named_modules()
        if name.startswith(prefix) and name.split(".")[-1] in suffixes and _is_linear(module)
    )
    if not names:
        raise ValueError(
            f"no Linear modules matched suffixes {sorted(suffixes)} within scope {scope!r}"
        )
    return names


def target_modules_digest(names: Sequence[str]) -> str:
    """Order-independent hash of the targeted module set."""
    payload = "\n".join(sorted(names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# --------------------------------------------------------------------------
# Adapter attachment
# --------------------------------------------------------------------------
def prepare_base_for_training(model, spec: MethodSpec, gradient_checkpointing: bool):
    """Method-specific preparation of the base model, and nothing more.

    QLoRA additionally runs ``prepare_model_for_kbit_training``, which upcasts
    layer norms to fp32 and makes the frozen 4-bit base compatible with
    gradient checkpointing. LoRA needs only the input-grad hook that gradient
    checkpointing requires. This asymmetry is inherent to QLoRA and is recorded
    in the run artifacts.
    """
    model.config.use_cache = False
    gc_kwargs = {"use_reentrant": False}

    if spec.prepare_for_kbit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=gc_kwargs,
        )
    elif gradient_checkpointing:
        model.enable_input_require_grads()
    return model


def attach_lora(model, cfg, target_modules: Sequence[str]):
    """Attach the LoRA adapter. Byte-identical configuration for both methods."""
    from peft import LoraConfig, get_peft_model

    lora_cfg = cfg.lora
    peft_config = LoraConfig(
        r=int(lora_cfg.r),
        lora_alpha=int(lora_cfg.alpha),
        lora_dropout=float(lora_cfg.dropout),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=list(target_modules),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)


def load_adapter(model, adapter_dir: str):
    """Load a trained adapter onto a base model for evaluation."""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)


def adapter_summary(cfg, target_modules: Sequence[str]) -> dict:
    """Adapter hyper-parameters, recorded so both runs can be diffed."""
    lora_cfg = cfg.lora
    return {
        "r": int(lora_cfg.r),
        "alpha": int(lora_cfg.alpha),
        "dropout": float(lora_cfg.dropout),
        "bias": str(lora_cfg.get("bias", "none")),
        "target_scope": lora_cfg.get("target_scope", "all"),
        "target_modules_suffixes": sorted(lora_cfg.target_modules_suffixes),
        "target_modules_resolved_count": len(target_modules),
        "target_modules_digest": target_modules_digest(target_modules),
        "target_modules_resolved": list(target_modules),
    }
