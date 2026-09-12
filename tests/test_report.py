"""Tests for the comparison assembly that the notebook consumes."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlm_ft import report

QUALITY = {
    "base": {"json_validity_rate": 0.62, "document_exact_match": 0.0, "field_accuracy": 0.11,
             "field_precision": 0.14, "field_recall": 0.12, "field_f1": 0.13,
             "cer": 0.88, "wer": 0.95, "eval_loss": 3.51},
    "lora": {"json_validity_rate": 1.0, "document_exact_match": 0.41, "field_accuracy": 0.86,
             "field_precision": 0.90, "field_recall": 0.88, "field_f1": 0.89,
             "cer": 0.12, "wer": 0.15, "eval_loss": 0.22},
    "qlora": {"json_validity_rate": 1.0, "document_exact_match": 0.37, "field_accuracy": 0.84,
              "field_precision": 0.88, "field_recall": 0.86, "field_f1": 0.87,
              "cer": 0.14, "wer": 0.17, "eval_loss": 0.25},
}
TOTAL = 2_246_251_008
TRAINABLE = 11_567_104


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _make_tree(root: Path, *, fingerprint_qlora="fp1", vram_qlora=3.1, include_qlora=True):
    for variant in ("base", "lora", "qlora"):
        if variant == "qlora" and not include_qlora:
            continue
        trainable = 0 if variant == "base" else TRAINABLE
        _write(root / "eval" / variant / "metrics.json", {
            "variant": variant,
            "metrics": {**QUALITY[variant], "num_documents": 100},
            "parameters": {"total": TOTAL, "trainable": trainable,
                           "trainable_ratio": trainable / TOTAL,
                           "trainable_percent": 100 * trainable / TOTAL,
                           "adapter_parameters": trainable},
            "inference": {"quantized_4bit": variant == "qlora"},
            "dataset": {"examples": 100},
            "model": {"id": "m"},
        })
        (root / "eval" / variant / "predictions.jsonl").write_text("{}\n")

    for variant, vram, seconds, fp in (("lora", 9.4, 3600.0, "fp1"), ("qlora", vram_qlora, 4320.0, fingerprint_qlora)):
        if variant == "qlora" and not include_qlora:
            continue
        _write(root / variant / "resource_metrics.json", {
            "method": variant,
            "gpu": {"name": "NVIDIA L4", "available": True, "total_vram_bytes": 23_836_360_704,
                    "total_vram_gib": 22.2,
                    "peak_memory_allocated_bytes": int(vram * 1024 ** 3),
                    "peak_memory_allocated_gib": vram,
                    "peak_memory_reserved_bytes": int((vram + 0.5) * 1024 ** 3),
                    "peak_memory_reserved_gib": vram + 0.5},
            "training": {"duration_seconds": seconds, "duration_human": "1h 00m 00s",
                         "steps": 200, "samples": 1600,
                         "steps_per_second": 200 / seconds, "samples_per_second": 1600 / seconds},
            "parameters": {"total": TOTAL, "trainable": TRAINABLE,
                           "trainable_ratio": TRAINABLE / TOTAL,
                           "trainable_percent": 100 * TRAINABLE / TOTAL,
                           "parameter_memory_gib": 4.2 if variant == "lora" else 1.5},
            "model_preparation": {"duration_seconds": 30.0, "peak_memory_allocated_gib": vram * 0.5},
            "quantization": {"enabled": variant == "qlora", "compute_dtype": "bfloat16"},
            "adapter": {"r": 16, "alpha": 32, "dropout": 0.05,
                        "target_modules_digest": "abc123", "target_modules_resolved_count": 168},
            "training_detail": {"optimizer": "adamw_torch", "lr_scheduler_type": "cosine",
                                "learning_rate": 2e-4, "effective_batch_size": 8,
                                "evaluation_seconds_within_training": 300.0,
                                "duration_seconds_excluding_evaluation": seconds - 300.0},
            "dataset": {"train_examples": 800},
            "config_fingerprint_excluding_method_keys": fp,
        })
        _write(root / variant / "training_log_history.json", [
            {"step": 1, "epoch": 0.01, "loss": 2.5},
            {"step": 25, "epoch": 0.25, "loss": 0.9},
            {"step": 25, "epoch": 0.25, "eval_loss": 0.8, "eval_runtime": 30.0},
            {"step": 200, "epoch": 2.0, "loss": 0.2},
        ])


def test_table_has_every_required_row(tmp_path):
    _make_tree(tmp_path)
    comparison = report.build_comparison(str(tmp_path))
    labels = [row["metric"] for row in comparison["table"]]
    for expected in ("JSON validity rate", "Document exact match", "Field accuracy",
                     "Field precision", "Field recall", "Field F1", "CER", "WER",
                     "Evaluation loss", "Peak training VRAM (GiB)", "Training time (s)",
                     "Total parameters", "Trainable parameters", "Trainable parameter ratio (%)"):
        assert expected in labels, expected


def test_base_resource_cells_are_not_applicable(tmp_path):
    _make_tree(tmp_path)
    table = {row["key"]: row for row in report.build_comparison(str(tmp_path))["table"]}
    assert table["peak_training_vram_gib"]["base"] is None
    assert table["training_duration_seconds"]["base"] is None
    assert table["trainable_parameters"]["base"] == 0
    assert table["total_parameters"]["base"] == TOTAL
    # and both fine-tuned variants do have them
    assert table["peak_training_vram_gib"]["lora"] == 9.4
    assert table["training_duration_seconds"]["qlora"] == 4320.0


def test_markdown_renders_na_for_base_training_rows(tmp_path):
    _make_tree(tmp_path)
    md = report.build_comparison(str(tmp_path))["markdown_table"]
    lines = {line.split("|")[1].strip(): line for line in md.splitlines() if line.startswith("|")}
    assert lines["Peak training VRAM (GiB)"].split("|")[2].strip() == "N/A"
    assert lines["Training time (s)"].split("|")[2].strip() == "N/A"
    assert lines["Trainable parameters"].split("|")[2].strip() == "0"
    assert "11,567,104" in lines["Trainable parameters"]


def test_analysis_quality_deltas(tmp_path):
    _make_tree(tmp_path)
    analysis = report.build_comparison(str(tmp_path))["analysis"]
    d = analysis["quality_deltas"]
    assert abs(d["lora_vs_base"]["field_f1"]["absolute"] - (0.89 - 0.13)) < 1e-9
    assert abs(d["qlora_vs_base"]["field_f1"]["absolute"] - (0.87 - 0.13)) < 1e-9
    assert abs(d["qlora_vs_lora"]["field_f1"]["absolute"] - (0.87 - 0.89)) < 1e-9
    # CER is lower-is-better; the delta is reported raw and the orientation lives in the table
    assert d["lora_vs_base"]["cer"]["absolute"] < 0


def test_analysis_vram_and_time(tmp_path):
    _make_tree(tmp_path)
    analysis = report.build_comparison(str(tmp_path))["analysis"]
    vram = analysis["vram"]
    assert vram["measured"] is True
    assert abs(vram["qlora_saving_gib"] - (9.4 - 3.1)) < 1e-3
    assert abs(vram["qlora_saving_percent"] - 100 * (9.4 - 3.1) / 9.4) < 1e-6
    timing = analysis["training_time"]
    assert timing["qlora_overhead_seconds"] == 720.0
    assert abs(timing["qlora_overhead_percent"] - 20.0) < 1e-9
    assert timing["lora_seconds_excluding_eval"] == 3300.0


def test_analysis_parameter_equality(tmp_path):
    _make_tree(tmp_path)
    params = report.build_comparison(str(tmp_path))["analysis"]["parameters"]
    assert params["trainable_identical"] is True
    assert params["total_identical"] is True
    assert params["base_trainable"] == 0
    # quantization changes footprint, not count
    assert params["lora_parameter_memory_gib"] > params["qlora_parameter_memory_gib"]


def test_controlled_comparison_passes_when_matched(tmp_path):
    _make_tree(tmp_path)
    check = report.build_comparison(str(tmp_path))["analysis"]["controlled_comparison"]
    assert check["checked"] and check["all_matched"], check.get("failed")
    assert len(check["documented_differences"]) >= 3


def test_controlled_comparison_flags_config_drift(tmp_path):
    _make_tree(tmp_path, fingerprint_qlora="DIFFERENT")
    check = report.build_comparison(str(tmp_path))["analysis"]["controlled_comparison"]
    assert check["checked"] and not check["all_matched"]
    assert "config_fingerprint_matches" in check["failed"]


def test_missing_qlora_degrades_to_na(tmp_path):
    _make_tree(tmp_path, include_qlora=False)
    comparison = report.build_comparison(str(tmp_path))
    table = {row["key"]: row for row in comparison["table"]}
    assert table["field_f1"]["qlora"] is None
    assert table["peak_training_vram_gib"]["qlora"] is None
    assert any("qlora" in p for p in comparison["missing_artifacts"])
    assert comparison["analysis"]["vram"]["measured"] is False
    assert comparison["analysis"]["controlled_comparison"]["checked"] is False
    assert "N/A" in comparison["markdown_table"]


def test_loss_curves_exclude_base(tmp_path):
    _make_tree(tmp_path)
    curves = report.load_loss_curves(str(tmp_path))
    assert set(curves) == {"lora", "qlora"}, "BASE must not get a training-loss curve"
    assert [p["loss"] for p in curves["lora"]["train"]] == [2.5, 0.9, 0.2]
    assert [p["loss"] for p in curves["lora"]["eval"]] == [0.8]


def test_csv_round_trips(tmp_path):
    _make_tree(tmp_path)
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(report.render_csv(report.build_comparison(str(tmp_path))["table"]))))
    by_key = {r["key"]: r for r in rows}
    assert by_key["field_f1"]["lora"] == "0.89"
    assert by_key["peak_training_vram_gib"]["base"] == ""


def test_format_value_treats_nan_as_not_measured():
    assert report.format_value("total_parameters", None) == "N/A"
    assert report.format_value("total_parameters", float("nan")) == "N/A"
    assert report.format_value("trainable_percent", float("nan")) == "N/A"
    assert report.format_value("total_parameters", 11567104) == "11,567,104"
    assert report.format_value("trainable_parameters", 0) == "0"
    assert report.format_value("trainable_percent", 0.5151) == "0.5151%"
    assert report.format_value("field_f1", 0.891234) == "0.8912"
    assert "1h 00m 00s" in report.format_value("training_duration_seconds", 3600.0)
