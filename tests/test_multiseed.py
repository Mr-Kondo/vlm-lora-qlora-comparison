"""Tests for multi-seed aggregation and the run_experiment orchestrator."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_ft import report  # noqa: E402

TOTAL = 2_246_251_008
TRAINABLE = 11_567_104

# seed -> variant -> field_f1 ; QLoRA is slightly behind LoRA in every seed
F1 = {
    42: {"base": 0.13, "lora": 0.89, "qlora": 0.87},
    43: {"base": 0.13, "lora": 0.91, "qlora": 0.88},
    44: {"base": 0.13, "lora": 0.87, "qlora": 0.86},
}
VRAM = {42: (9.4, 3.1), 43: (9.5, 3.2), 44: (9.3, 3.0)}
SECONDS = {42: (3600.0, 4320.0), 43: (3660.0, 4400.0), 44: (3540.0, 4260.0)}


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _make_seed(root: Path, seed: int, gpu="NVIDIA A100-SXM4-40GB", include_qlora=True):
    for variant in ("base", "lora", "qlora"):
        if variant == "qlora" and not include_qlora:
            continue
        trainable = 0 if variant == "base" else TRAINABLE
        f1 = F1[seed][variant]
        _write(root / "eval" / variant / "metrics.json", {
            "variant": variant,
            "metrics": {
                "json_validity_rate": 1.0 if variant != "base" else 0.62,
                "document_exact_match": f1 - 0.45,
                "field_accuracy": f1 - 0.02,
                "field_precision": f1 + 0.01,
                "field_recall": f1 - 0.01,
                "field_f1": f1,
                "cer": 1.0 - f1,
                "wer": 1.05 - f1,
                "eval_loss": 3.5 - 3.2 * f1,
                "num_documents": 100,
            },
            "parameters": {"total": TOTAL, "trainable": trainable,
                           "trainable_ratio": trainable / TOTAL,
                           "trainable_percent": 100 * trainable / TOTAL},
            "dataset": {"examples": 100},
        })
        (root / "eval" / variant / "predictions.jsonl").write_text("{}\n")

    for index, method in enumerate(("lora", "qlora")):
        if method == "qlora" and not include_qlora:
            continue
        vram, seconds = VRAM[seed][index], SECONDS[seed][index]
        _write(root / method / "resource_metrics.json", {
            "method": method,
            "gpu": {"name": gpu, "available": True, "total_vram_bytes": 42_949_672_960,
                    "total_vram_gib": 40.0,
                    "peak_memory_allocated_bytes": int(vram * 1024 ** 3),
                    "peak_memory_allocated_gib": vram,
                    "peak_memory_reserved_gib": vram + 0.5},
            "training": {"duration_seconds": seconds, "duration_human": "1h", "steps": 200,
                         "samples": 1600},
            "parameters": {"total": TOTAL, "trainable": TRAINABLE,
                           "trainable_ratio": TRAINABLE / TOTAL,
                           "trainable_percent": 100 * TRAINABLE / TOTAL},
            "model_preparation": {"duration_seconds": 30.0},
            "quantization": {"enabled": method == "qlora", "compute_dtype": "bfloat16"},
            "adapter": {"r": 16, "alpha": 32, "dropout": 0.05,
                        "target_modules_digest": "abc123", "target_modules_resolved_count": 168},
            "training_detail": {"optimizer": "adamw_torch", "lr_scheduler_type": "cosine",
                                "learning_rate": 2e-4, "effective_batch_size": 8},
            "dataset": {"train_examples": 800},
            "config_fingerprint_excluding_method_keys": "fp1",
        })
        _write(root / method / "training_log_history.json",
               [{"step": 1, "loss": 2.5}, {"step": 200, "loss": 0.2},
                {"step": 200, "eval_loss": 0.3, "eval_runtime": 20.0}])


def _make_study(tmp_path: Path, seeds=(42, 43, 44), **kwargs):
    for seed in seeds:
        _make_seed(Path(report.seed_root(str(tmp_path), seed)), seed, **kwargs)
    return tmp_path


# ------------------------------------------------------------------ layout
def test_seed_root_layout():
    assert report.seed_root("outputs", 42).endswith("outputs/seed42")


def test_finds_every_seed(tmp_path):
    _make_study(tmp_path)
    c = report.build_multi_seed_comparison(str(tmp_path), [42, 43, 44])
    assert c["seeds_found"] == [42, 43, 44]
    assert c["seeds_missing"] == []


def test_reports_missing_seeds_without_inventing_them(tmp_path):
    _make_study(tmp_path, seeds=(42, 43))
    c = report.build_multi_seed_comparison(str(tmp_path), [42, 43, 44])
    assert c["seeds_found"] == [42, 43]
    assert c["seeds_missing"] == [44]
    f1 = next(r for r in c["table"] if r["key"] == "field_f1")
    assert f1["lora"]["n"] == 2


# ------------------------------------------------------------ statistics
def test_mean_and_std_across_seeds(tmp_path):
    _make_study(tmp_path)
    table = {r["key"]: r for r in report.build_multi_seed_comparison(str(tmp_path), [42, 43, 44])["table"]}
    lora = table["field_f1"]["lora"]
    assert lora["n"] == 3
    assert abs(lora["mean"] - (0.89 + 0.91 + 0.87) / 3) < 1e-12
    assert lora["min"] == 0.87 and lora["max"] == 0.91
    assert lora["std"] == pytest.approx(0.02, abs=1e-9)   # sample std of .89/.91/.87


def test_single_seed_reports_no_std(tmp_path):
    _make_study(tmp_path, seeds=(42,))
    table = {r["key"]: r for r in report.build_multi_seed_comparison(str(tmp_path), [42])["table"]}
    stats = table["field_f1"]["lora"]
    assert stats["n"] == 1
    assert stats["std"] is None, "one observation must not look like zero variance"
    assert "n=1" in report.format_stats("field_f1", stats)


def test_base_rows_stay_correct_across_seeds(tmp_path):
    _make_study(tmp_path)
    table = {r["key"]: r for r in report.build_multi_seed_comparison(str(tmp_path), [42, 43, 44])["table"]}
    assert table["trainable_parameters"]["base"]["mean"] == 0
    assert table["total_parameters"]["base"]["mean"] == TOTAL
    # BASE has no training run, so its resource cells stay unmeasured
    assert table["peak_training_vram_gib"]["base"]["n"] == 0
    assert table["training_duration_seconds"]["base"]["n"] == 0


def test_trainable_parameters_identical_across_methods_and_seeds(tmp_path):
    _make_study(tmp_path)
    table = {r["key"]: r for r in report.build_multi_seed_comparison(str(tmp_path), [42, 43, 44])["table"]}
    for method in ("lora", "qlora"):
        assert table["trainable_parameters"][method]["std"] == 0.0
        assert table["trainable_parameters"][method]["mean"] == TRAINABLE


# --------------------------------------------------------------- pairing
def test_paired_difference_and_sign_consistency(tmp_path):
    _make_study(tmp_path)
    paired = report.build_multi_seed_comparison(str(tmp_path), [42, 43, 44])["paired_lora_vs_qlora"]
    f1 = paired["metrics"]["field_f1"]
    assert [e["seed"] for e in f1["per_seed"]] == [42, 43, 44]
    expected = [(0.87 - 0.89), (0.88 - 0.91), (0.86 - 0.87)]
    assert f1["difference"]["mean"] == pytest.approx(sum(expected) / 3)
    assert f1["seeds_favouring_baseline"] == 3      # LoRA wins every seed
    assert f1["seeds_favouring_candidate"] == 0
    assert f1["sign_is_consistent"] is True


def test_pairing_respects_metric_orientation(tmp_path):
    """CER is lower-is-better: a positive difference must count against QLoRA."""
    _make_study(tmp_path)
    paired = report.build_multi_seed_comparison(str(tmp_path), [42, 43, 44])["paired_lora_vs_qlora"]
    cer = paired["metrics"]["cer"]
    assert cer["difference"]["mean"] > 0, "fixture has QLoRA with the higher CER"
    assert cer["seeds_favouring_baseline"] == 3
    assert cer["seeds_favouring_candidate"] == 0


def test_pairing_handles_a_missing_method(tmp_path):
    _make_study(tmp_path, include_qlora=False)
    paired = report.build_multi_seed_comparison(str(tmp_path), [42, 43, 44])["paired_lora_vs_qlora"]
    f1 = paired["metrics"]["field_f1"]
    assert f1["per_seed"] == [] and f1["difference"]["n"] == 0
    assert f1["sign_is_consistent"] is False


# ------------------------------------------------------- controlled check
def test_controlled_check_passes_across_seeds(tmp_path):
    _make_study(tmp_path)
    check = report.build_multi_seed_comparison(str(tmp_path), [42, 43, 44])["controlled_comparison"]
    assert check["seeds_checked"] == [42, 43, 44]
    assert check["all_seeds_matched"] is True
    assert check["same_gpu_across_seeds"] is True
    assert check["gpu_warning"] is None


def test_controlled_check_flags_a_gpu_change_between_seeds(tmp_path):
    for seed in (42, 43):
        _make_seed(Path(report.seed_root(str(tmp_path), seed)), seed)
    _make_seed(Path(report.seed_root(str(tmp_path), 44)), 44, gpu="Tesla T4")
    check = report.build_multi_seed_comparison(str(tmp_path), [42, 43, 44])["controlled_comparison"]
    assert check["same_gpu_across_seeds"] is False
    assert "Tesla T4" in check["gpu_warning"]


def test_unchecked_is_none_not_false(tmp_path):
    _make_study(tmp_path, include_qlora=False)
    check = report.build_multi_seed_comparison(str(tmp_path), [42, 43, 44])["controlled_comparison"]
    assert check["seeds_checked"] == []
    assert check["all_seeds_matched"] is None, "'could not check' must differ from 'did not match'"


# --------------------------------------------------------------- rendering
def test_markdown_shows_mean_and_spread(tmp_path):
    _make_study(tmp_path)
    md = report.build_multi_seed_comparison(str(tmp_path), [42, 43, 44])["markdown_table"]
    assert "±" in md
    rows = {line.split("|")[1].strip(): line for line in md.splitlines() if line.startswith("|")}
    assert rows["Peak training VRAM (GiB)"].split("|")[2].strip() == "N/A"   # BASE
    assert rows["Trainable parameters"].split("|")[2].strip().startswith("0")


def test_csv_carries_per_seed_values(tmp_path):
    import csv
    import io

    _make_study(tmp_path)
    c = report.build_multi_seed_comparison(str(tmp_path), [42, 43, 44])
    rows = list(csv.DictReader(io.StringIO(report.render_multi_seed_csv(c["table"], [42, 43, 44]))))
    f1 = next(r for r in rows if r["key"] == "field_f1")
    assert float(f1["lora_mean"]) == pytest.approx(0.89)
    assert [float(f1[f"seed_{s}_lora"]) for s in (42, 43, 44)] == [0.89, 0.91, 0.87]


def test_loss_curves_are_per_seed_and_exclude_base(tmp_path):
    _make_study(tmp_path)
    curves = report.load_multi_seed_loss_curves(str(tmp_path), [42, 43, 44])
    assert sorted(curves) == [42, 43, 44]
    for seed_curves in curves.values():
        assert set(seed_curves) == {"lora", "qlora"}


# -------------------------------------------------------------- orchestrator
def _runner():
    spec = importlib.util.spec_from_file_location("_run_experiment", ROOT / "scripts" / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_builds_matched_commands():
    runner = _runner()
    train = runner.build_command("train", "qlora", 43, "outputs/seed43", [])
    assert "--method" in train and "qlora" in train
    assert "experiment.seed=43" in train
    assert "experiment.output_root=outputs/seed43" in train

    evaluate = runner.build_command("evaluate", "base", 43, "outputs/seed43", [])
    assert "--model-variant" in evaluate and "base" in evaluate


def test_runner_passes_overrides_to_every_step():
    runner = _runner()
    for kind, name in (("train", "lora"), ("evaluate", "qlora")):
        command = runner.build_command(kind, name, 42, "out", ["data.image.longest_edge=1536"])
        assert "data.image.longest_edge=1536" in command


def test_runner_artifact_paths_match_the_report_layout(tmp_path):
    runner = _runner()
    _make_study(tmp_path, seeds=(42,))
    root = report.seed_root(str(tmp_path), 42)
    assert Path(runner.artifact_for("train", "lora", root)).is_file()
    assert Path(runner.artifact_for("evaluate", "base", root)).is_file()
    assert not Path(runner.artifact_for("train", "nope", root)).is_file()


def test_runner_plans_the_full_matrix(capsys):
    runner = _runner()
    assert runner.main(["--seeds", "42", "43", "44", "--output-root", "/tmp/nonexistent-x", "--dry-run"]) == 0
    printed = capsys.readouterr().out
    # 3 seeds x (2 trainings + 3 evaluations)
    assert printed.count("+ ") == 15
    assert "15 steps would run" in printed


def test_an_all_tie_is_not_reported_as_consistent(tmp_path):
    """Every seed tying means there is no direction, not a consistent one."""
    _make_study(tmp_path)
    paired = report.build_multi_seed_comparison(str(tmp_path), [42, 43, 44])["paired_lora_vs_qlora"]
    # the fixture gives both methods a JSON validity rate of 1.0 in every seed
    validity = paired["metrics"]["json_validity_rate"]
    assert validity["difference"]["mean"] == 0.0
    assert validity["seeds_tied"] == 3
    assert validity["all_tied"] is True
    assert validity["sign_is_consistent"] is False
    # while a genuine one-sided result still is consistent
    assert paired["metrics"]["field_f1"]["sign_is_consistent"] is True
    assert paired["metrics"]["field_f1"]["all_tied"] is False
