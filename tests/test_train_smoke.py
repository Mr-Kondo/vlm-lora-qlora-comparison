"""End-to-end smoke test for scripts/train.py on CPU with a tiny stand-in model.

Exercises the real shared pipeline -- config loading, target resolution, adapter
attachment, the collator, the Trainer loop and every artifact write -- without
needing a GPU. The QLoRA branch additionally needs bitsandbytes + CUDA and is
skipped where that is unavailable.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TINY = "hf-internal-testing/tiny-random-Idefics3ForConditionalGeneration"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_split(n, seed=0):
    """CORD-shaped records: a PIL image plus a ground_truth JSON string."""
    import random

    from datasets import Dataset
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    rows = {"image": [], "ground_truth": []}
    for i in range(n):
        img = Image.new("RGB", (192, 288), "white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10 + i), f"ITEM {i}", fill="black")
        price = f"{rng.randint(1, 40)}.000"
        rows["image"].append(img)
        rows["ground_truth"].append(
            json.dumps({
                "gt_parse": {
                    "menu": {"nm": f"ITEM {i}", "price": price},
                    "total": {"total_price": price},
                },
                "meta": {"image_id": i},
            })
        )
    return Dataset.from_dict(rows).cast_column("image", __import__("datasets").Image())


@pytest.fixture()
def patched_splits(monkeypatch):
    from vlm_ft import data as data_mod

    def fake(cfg, splits=("train", "validation", "test")):
        sizes = {"train": 4, "validation": 2, "test": 2}
        return {k: _synthetic_split(sizes[k], seed=hash(k) % 100) for k in splits}

    monkeypatch.setattr(data_mod, "load_splits", fake)
    return fake


TINY_OVERRIDES = [
    "--set", f"model.id={TINY}",
    "--set", "model.compute_dtype=float32",
    "--set", "model.attn_implementation=eager",
    "--set", "data.max_seq_length=4096",
    "--set", "training.max_steps=2",
    "--set", "training.gradient_accumulation_steps=1",
    "--set", "training.eval_steps=2",
    "--set", "training.logging_steps=1",
    "--set", "training.dataloader_num_workers=0",
    "--set", "lora.r=4",
    "--set", "lora.alpha=8",
]


def test_lora_training_produces_complete_artifacts(tmp_path, patched_splits):
    train = _load_script("train")
    out = tmp_path / "lora"
    assert train.main([
        "--method", "lora",
        "--config", str(ROOT / "configs" / "lora.yaml"),
        "--output-dir", str(out),
        *TINY_OVERRIDES,
    ]) == 0

    # adapter is loadable
    assert (out / "adapter" / "adapter_config.json").is_file()
    adapter_cfg = json.loads((out / "adapter" / "adapter_config.json").read_text())
    assert adapter_cfg["r"] == 4 and adapter_cfg["lora_alpha"] == 8

    rm = json.loads((out / "resource_metrics.json").read_text())
    assert rm["method"] == "lora"
    for key in ("name", "total_vram_bytes", "peak_memory_allocated_bytes", "peak_memory_reserved_bytes"):
        assert key in rm["gpu"]
    for key in ("duration_seconds", "steps", "samples", "steps_per_second", "samples_per_second"):
        assert key in rm["training"]
    for key in ("total", "trainable", "trainable_ratio"):
        assert key in rm["parameters"]

    assert rm["training"]["duration_seconds"] > 0
    assert rm["training"]["steps"] == 2
    assert rm["parameters"]["trainable"] > 0
    assert rm["parameters"]["total"] > rm["parameters"]["trainable"]
    assert 0 < rm["parameters"]["trainable_ratio"] < 1
    assert rm["quantization"]["enabled"] is False
    assert rm["quantization"]["prepared_for_kbit_training"] is False
    # only the language tower is adapted
    assert all(m.startswith("model.text_model.") for m in rm["adapter"]["target_modules_resolved"])
    # training time excludes model load/quantization
    assert rm["model_preparation"]["duration_seconds"] > 0

    history = json.loads((out / "training_log_history.json").read_text())
    assert any("loss" in e for e in history), "training loss was never logged"
    assert any("eval_loss" in e for e in history), "validation loss was never logged"

    run_cfg = json.loads((out / "run_config.json").read_text())
    assert run_cfg["resolved_config"]["lora"]["target_scope"] == "text_model"


def test_method_and_config_mismatch_is_rejected(tmp_path, patched_splits):
    train = _load_script("train")
    with pytest.raises(ValueError, match="config declares method"):
        train.main([
            "--method", "lora",
            "--config", str(ROOT / "configs" / "qlora.yaml"),
            "--output-dir", str(tmp_path / "x"),
            *TINY_OVERRIDES,
        ])


def test_quantization_flag_must_match_method(tmp_path, patched_splits):
    train = _load_script("train")
    with pytest.raises(ValueError, match="requires quantization.load_in_4bit"):
        train.main([
            "--method", "lora",
            "--config", str(ROOT / "configs" / "lora.yaml"),
            "--output-dir", str(tmp_path / "x"),
            "--set", "quantization.load_in_4bit=true",
            *TINY_OVERRIDES,
        ])
