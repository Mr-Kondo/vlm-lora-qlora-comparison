"""Tests for config inheritance, overrides and the LoRA/QLoRA method switch."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_ft import modeling  # noqa: E402
from vlm_ft.config import ConfigError, config_fingerprint, load_config  # noqa: E402

CONFIGS = ROOT / "configs"
SHARED_KEYS = ("experiment", "model", "data", "prompt", "training", "lora", "generation", "evaluation")


def test_both_method_configs_inherit_the_same_base():
    lora = load_config(CONFIGS / "lora.yaml")
    qlora = load_config(CONFIGS / "qlora.yaml")
    for section in SHARED_KEYS:
        assert lora[section].to_dict() == qlora[section].to_dict(), (
            f"{section} differs between lora.yaml and qlora.yaml; the comparison would be confounded"
        )


def test_only_quantization_and_method_differ():
    lora = load_config(CONFIGS / "lora.yaml").to_dict()
    qlora = load_config(CONFIGS / "qlora.yaml").to_dict()
    differing = {k for k in set(lora) | set(qlora) if lora.get(k) != qlora.get(k)}
    assert differing <= {"method", "quantization", "_config_path"}, differing


def test_fingerprint_matches_when_method_keys_are_excluded():
    exclude = ("method", "quantization", "_config_path", "_parent_config")
    assert config_fingerprint(load_config(CONFIGS / "lora.yaml"), exclude) == config_fingerprint(
        load_config(CONFIGS / "qlora.yaml"), exclude
    )


def test_fingerprint_detects_a_real_difference():
    exclude = ("method", "quantization", "_config_path", "_parent_config")
    a = config_fingerprint(load_config(CONFIGS / "lora.yaml"), exclude)
    b = config_fingerprint(load_config(CONFIGS / "lora.yaml", ["training.learning_rate=1e-5"]), exclude)
    assert a != b


def test_overrides_parse_yaml_scalars():
    cfg = load_config(CONFIGS / "lora.yaml", [
        "training.num_train_epochs=3",
        "data.max_train_samples=null",
        "training.gradient_checkpointing=false",
        "model.id=some/model",
        "data.image.longest_edge=1152",
    ])
    assert cfg.training.num_train_epochs == 3
    assert cfg.data.get("max_train_samples") is None
    assert cfg.training.gradient_checkpointing is False
    assert cfg.model.id == "some/model"
    assert cfg.data.image.longest_edge == 1152


def test_missing_key_raises_instead_of_returning_none():
    cfg = load_config(CONFIGS / "lora.yaml")
    with pytest.raises(ConfigError, match="training.nonexistent"):
        _ = cfg.training.nonexistent


def test_malformed_override_rejected():
    with pytest.raises(ConfigError, match="key.path=value"):
        load_config(CONFIGS / "lora.yaml", ["training.learning_rate"])


def test_method_specs_differ_only_where_intended():
    lora, qlora = modeling.METHOD_SPECS["lora"], modeling.METHOD_SPECS["qlora"]
    assert (lora.quantize_base, lora.prepare_for_kbit) == (False, False)
    assert (qlora.quantize_base, qlora.prepare_for_kbit) == (True, True)


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        modeling.get_method_spec("full-finetune")


def test_qlora_quantization_config_is_nf4_double_quant():
    import torch

    cfg = load_config(CONFIGS / "qlora.yaml")
    spec = modeling.validate_method_against_config("qlora", cfg)
    quant = modeling.build_quantization_config(cfg, spec, torch.bfloat16)
    assert quant.load_in_4bit is True
    assert quant.bnb_4bit_quant_type == "nf4"
    assert quant.bnb_4bit_use_double_quant is True
    # compute dtype follows the shared setting, so both methods compute alike
    assert quant.bnb_4bit_compute_dtype == torch.bfloat16


def test_lora_has_no_quantization_config():
    import torch

    cfg = load_config(CONFIGS / "lora.yaml")
    spec = modeling.validate_method_against_config("lora", cfg)
    assert modeling.build_quantization_config(cfg, spec, torch.bfloat16) is None


def test_compute_dtype_resolution():
    import torch

    assert modeling.resolve_compute_dtype("bfloat16") is torch.bfloat16
    assert modeling.resolve_compute_dtype("fp16") is torch.float16
    assert modeling.resolve_compute_dtype("float32") is torch.float32
    assert modeling.resolve_compute_dtype("auto") in (torch.bfloat16, torch.float16, torch.float32)
    with pytest.raises(ValueError, match="unsupported dtype"):
        modeling.resolve_compute_dtype("int4")


def test_target_modules_digest_is_order_independent():
    a = ["m.layers.0.q_proj", "m.layers.1.k_proj"]
    assert modeling.target_modules_digest(a) == modeling.target_modules_digest(list(reversed(a)))
    assert modeling.target_modules_digest(a) != modeling.target_modules_digest(a + ["m.lm_head"])
