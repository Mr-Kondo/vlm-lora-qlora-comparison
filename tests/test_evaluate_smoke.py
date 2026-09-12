"""End-to-end smoke test for scripts/evaluate.py on CPU with a tiny model.

Covers the BASE path (no adapter) and the LoRA path (adapter trained by the
companion smoke test), including artifact schema and the "BASE has 0 trainable
parameters / LoRA reports its adapter size" rule.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from test_train_smoke import TINY, TINY_OVERRIDES, _load_script, patched_splits  # noqa: F401,E402

EVAL_OVERRIDES = [
    "--set", f"model.id={TINY}",
    "--set", "model.compute_dtype=float32",
    "--set", "model.attn_implementation=eager",
    "--set", "data.max_seq_length=4096",
    "--set", "generation.max_new_tokens=6",
    "--set", "evaluation.batch_size=1",
]


def _read(path):
    return json.loads(Path(path).read_text())


def test_base_variant_reports_zero_trainable_parameters(tmp_path, patched_splits):  # noqa: F811
    evaluate = _load_script("evaluate")
    out = tmp_path / "eval_base"
    assert evaluate.main([
        "--model-variant", "base",
        "--config", str(ROOT / "configs" / "base.yaml"),
        "--output-dir", str(out),
        *EVAL_OVERRIDES,
    ]) == 0

    payload = _read(out / "metrics.json")
    assert payload["variant"] == "base"
    assert payload["parameters"]["trainable"] == 0
    assert payload["parameters"]["trainable_ratio"] == 0.0
    assert payload["parameters"]["total"] > 0
    assert payload["inference"]["quantized_4bit"] is False
    assert payload["model"]["adapter_dir"] is None

    m = payload["metrics"]
    for key in ("json_validity_rate", "document_exact_match", "field_accuracy",
                "field_precision", "field_recall", "field_f1", "cer", "wer"):
        assert key in m, key
        assert isinstance(m[key], (int, float))
    assert m["num_documents"] == 2
    assert m["eval_loss"] is not None and m["eval_loss"] > 0

    lines = (out / "predictions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert {"prediction_raw", "prediction_canonical", "reference", "json_valid"} <= set(record)


def test_lora_variant_reports_adapter_parameters(tmp_path, patched_splits):  # noqa: F811
    train = _load_script("train")
    train_out = tmp_path / "lora"
    assert train.main([
        "--method", "lora",
        "--config", str(ROOT / "configs" / "lora.yaml"),
        "--output-dir", str(train_out),
        *TINY_OVERRIDES,
    ]) == 0
    trained = _read(train_out / "resource_metrics.json")

    evaluate = _load_script("evaluate")
    out = tmp_path / "eval_lora"
    assert evaluate.main([
        "--model-variant", "lora",
        "--config", str(ROOT / "configs" / "lora.yaml"),
        "--adapter-dir", str(train_out / "adapter"),
        "--output-dir", str(out),
        *EVAL_OVERRIDES,
    ]) == 0

    payload = _read(out / "metrics.json")
    assert payload["parameters"]["trainable"] > 0, "a fine-tuned variant must not report 0"
    assert payload["parameters"]["adapter_parameters"] == payload["parameters"]["trainable"]
    # the adapter counted at evaluation time matches what training reported
    assert payload["parameters"]["trainable"] == trained["parameters"]["trainable"]
    assert payload["metrics"]["eval_loss"] is not None


def test_missing_adapter_is_a_clear_error(tmp_path, patched_splits):  # noqa: F811
    evaluate = _load_script("evaluate")
    with pytest.raises(FileNotFoundError, match="adapter directory not found"):
        evaluate.main([
            "--model-variant", "lora",
            "--config", str(ROOT / "configs" / "lora.yaml"),
            "--adapter-dir", str(tmp_path / "nope"),
            "--output-dir", str(tmp_path / "x"),
            *EVAL_OVERRIDES,
        ])


def test_quantization_resolution():
    evaluate = _load_script("evaluate")
    assert evaluate.resolve_quantization("base", "auto") is False
    assert evaluate.resolve_quantization("lora", "auto") is False
    assert evaluate.resolve_quantization("qlora", "auto") is True
    assert evaluate.resolve_quantization("qlora", "none") is False
    assert evaluate.resolve_quantization("lora", "4bit") is True
