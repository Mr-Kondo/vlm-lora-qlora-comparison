"""Assemble the BASE / LoRA / QLoRA comparison from the on-disk artifacts.

Every number here is read from a JSON artifact written by ``scripts/train.py``
or ``scripts/evaluate.py``. Nothing is estimated: a metric that was not measured
comes back as ``None`` and is rendered as ``N/A``.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

from . import metrics as metrics_mod
from .resources import BYTES_PER_GIB, format_duration, read_json

VARIANTS = ("base", "lora", "qlora")

#: Rows of the final comparison table: (key, label, higher_is_better, group).
#: ``higher_is_better=None`` means the value is descriptive, not a score.
TABLE_ROWS: tuple[tuple[str, str, bool | None, str], ...] = (
    ("json_validity_rate", "JSON validity rate", True, "quality"),
    ("document_exact_match", "Document exact match", True, "quality"),
    ("field_accuracy", "Field accuracy", True, "quality"),
    ("field_precision", "Field precision", True, "quality"),
    ("field_recall", "Field recall", True, "quality"),
    ("field_f1", "Field F1", True, "quality"),
    ("cer", "CER", False, "quality"),
    ("wer", "WER", False, "quality"),
    ("eval_loss", "Evaluation loss", False, "quality"),
    ("peak_training_vram_gib", "Peak training VRAM (GiB)", False, "resource"),
    ("training_duration_seconds", "Training time (s)", False, "resource"),
    ("total_parameters", "Total parameters", None, "parameters"),
    ("trainable_parameters", "Trainable parameters", None, "parameters"),
    ("trainable_percent", "Trainable parameter ratio (%)", None, "parameters"),
)

#: Differences between LoRA and QLoRA that cannot be removed, with the reason.
#: Everything not listed here is held identical by the shared config and code.
DOCUMENTED_DIFFERENCES: tuple[dict[str, str], ...] = (
    {
        "parameter": "base weight storage",
        "lora": "16-bit compute dtype (bfloat16 or float16)",
        "qlora": "4-bit NF4 with double quantization",
        "why": "this is the definition of QLoRA; it is the independent variable",
    },
    {
        "parameter": "prepare_model_for_kbit_training",
        "lora": "not applied (only enable_input_require_grads for checkpointing)",
        "qlora": "applied: casts layer norms to fp32 and makes the frozen 4-bit base "
                 "compatible with gradient checkpointing",
        "why": "required for a quantized base to train at all; has no meaning for an "
               "unquantized model",
    },
    {
        "parameter": "evaluation-time weight precision",
        "lora": "16-bit base + adapter",
        "qlora": "4-bit base + adapter (the deployable form), overridable with "
                 "--quantization none",
        "why": "QLoRA's adapters are trained against quantized base weights, so the "
               "realistic deployment keeps them quantized; the flag exists to separate "
               "the training effect from the inference effect",
    },
    {
        "parameter": "evaluation loss comparability",
        "lora": "forward pass through 16-bit weights",
        "qlora": "forward pass through 4-bit weights",
        "why": "the loss values are computed by identical code but not through identical "
               "arithmetic, so small differences are expected independent of adapter quality",
    },
)


def artifact_paths(output_root: str, variant: str) -> dict:
    return {
        "eval_metrics": os.path.join(output_root, "eval", variant, "metrics.json"),
        "predictions": os.path.join(output_root, "eval", variant, "predictions.jsonl"),
        "resource_metrics": os.path.join(output_root, variant, "resource_metrics.json"),
        "training_log": os.path.join(output_root, variant, "training_log_history.json"),
    }


def load_variant(output_root: str, variant: str) -> dict:
    """Read one variant's artifacts. Missing files degrade to ``None`` values."""
    paths = artifact_paths(output_root, variant)
    record: dict[str, Any] = {
        "variant": variant,
        "paths": paths,
        "eval_available": os.path.isfile(paths["eval_metrics"]),
        "training_available": variant != "base" and os.path.isfile(paths["resource_metrics"]),
        "quality": {key: None for key, _, _ in metrics_mod.HEADLINE_METRICS},
        "resources": {},
        "parameters": {},
    }

    if record["eval_available"]:
        payload = read_json(paths["eval_metrics"])
        measured = payload.get("metrics", {})
        record["quality"] = {key: measured.get(key) for key, _, _ in metrics_mod.HEADLINE_METRICS}
        record["quality_full"] = measured
        record["parameters"] = dict(payload.get("parameters", {}))
        record["inference"] = payload.get("inference", {})
        record["eval_loss_detail"] = payload.get("eval_loss_detail", {})
        record["dataset"] = payload.get("dataset", {})
        record["model"] = payload.get("model", {})

    if variant == "base":
        # BASE is not fine-tuned: training resources are not applicable, not zero.
        record["resources"] = {
            "applicable": False,
            "reason": "BASE is evaluated without fine-tuning, so it has no training run",
        }
        record["parameters"].setdefault("trainable", 0)
        record["parameters"].setdefault("trainable_ratio", 0.0)
        record["parameters"].setdefault("trainable_percent", 0.0)
        return record

    if record["training_available"]:
        payload = read_json(paths["resource_metrics"])
        gpu = payload.get("gpu", {})
        training = payload.get("training", {})
        detail = payload.get("training_detail", {})
        record["resources"] = {
            "applicable": True,
            "gpu_name": gpu.get("name"),
            "gpu_available": gpu.get("available"),
            "gpu_unavailable_reason": gpu.get("unavailable_reason"),
            "total_vram_bytes": gpu.get("total_vram_bytes"),
            "total_vram_gib": gpu.get("total_vram_gib"),
            "peak_training_vram_bytes": gpu.get("peak_memory_allocated_bytes"),
            "peak_training_vram_gib": gpu.get("peak_memory_allocated_gib"),
            "peak_training_vram_reserved_bytes": gpu.get("peak_memory_reserved_bytes"),
            "peak_training_vram_reserved_gib": gpu.get("peak_memory_reserved_gib"),
            "training_duration_seconds": training.get("duration_seconds"),
            "training_duration_human": training.get("duration_human"),
            "steps": training.get("steps"),
            "samples": training.get("samples"),
            "steps_per_second": training.get("steps_per_second"),
            "samples_per_second": training.get("samples_per_second"),
            "model_preparation_seconds": payload.get("model_preparation", {}).get("duration_seconds"),
            "model_load_peak_vram_gib": payload.get("model_preparation", {}).get(
                "peak_memory_allocated_gib"
            ),
            "evaluation_seconds_within_training": detail.get("evaluation_seconds_within_training"),
            "training_duration_excluding_evaluation": detail.get(
                "duration_seconds_excluding_evaluation"
            ),
        }
        # Training-time counts are authoritative for trainable parameters.
        record["training_parameters"] = payload.get("parameters", {})
        record["adapter"] = payload.get("adapter", {})
        record["quantization"] = payload.get("quantization", {})
        record["training_detail"] = detail
        record["config_fingerprint"] = payload.get("config_fingerprint_excluding_method_keys")
        record["environment"] = payload.get("environment", {})
        record["dataset_training"] = payload.get("dataset", {})
        if not record["parameters"]:
            record["parameters"] = dict(payload.get("parameters", {}))
    else:
        record["resources"] = {
            "applicable": True,
            "reason": f"no training artifact at {paths['resource_metrics']}",
        }
    return record


def load_all(output_root: str, variants: Iterable[str] = VARIANTS) -> dict:
    return {variant: load_variant(output_root, variant) for variant in variants}


#: Table key -> key inside the variant record's ``parameters`` section.
_PARAMETER_KEYS = {
    "total_parameters": "total",
    "trainable_parameters": "trainable",
    "trainable_percent": "trainable_percent",
}


def _row_value(record: dict, key: str, group: str):
    """Look a table key up in the section it belongs to.

    The group drives the lookup rather than a search across sections, so BASE
    keeps its parameter counts (total measured, trainable 0 by definition) while
    only its *training* resource cells collapse to N/A.
    """
    if group == "quality":
        return record.get("quality", {}).get(key)
    if group == "resource":
        resources_section = record.get("resources", {})
        if not resources_section.get("applicable", True):
            return None
        return resources_section.get(key)
    if group == "parameters":
        return record.get("parameters", {}).get(_PARAMETER_KEYS[key])
    raise ValueError(f"unknown table group: {group!r}")


def build_table(variants: dict) -> list[dict]:
    """The final comparison table as records, ``None`` wherever unmeasured."""
    rows = []
    for key, label, higher_is_better, group in TABLE_ROWS:
        row = {"key": key, "metric": label, "higher_is_better": higher_is_better, "group": group}
        for variant in VARIANTS:
            row[variant] = (
                _row_value(variants[variant], key, group) if variant in variants else None
            )
        rows.append(row)
    return rows


def _delta(new, old):
    """Absolute and relative change, guarding against missing or zero baselines."""
    if new is None or old is None:
        return {"absolute": None, "relative": None}
    absolute = new - old
    relative = (absolute / abs(old)) if old else None
    return {"absolute": absolute, "relative": relative}


def check_controlled_comparison(variants: dict) -> dict:
    """Verify that the two fine-tuning runs really were matched."""
    lora, qlora = variants.get("lora", {}), variants.get("qlora", {})
    if not (lora.get("training_available") and qlora.get("training_available")):
        return {"checked": False, "reason": "both training artifacts are required"}

    lora_adapter = lora.get("adapter", {})
    qlora_adapter = qlora.get("adapter", {})
    checks = {
        "config_fingerprint_matches": lora.get("config_fingerprint") == qlora.get("config_fingerprint"),
        "target_modules_match": lora_adapter.get("target_modules_digest")
        == qlora_adapter.get("target_modules_digest"),
        "target_module_count_matches": lora_adapter.get("target_modules_resolved_count")
        == qlora_adapter.get("target_modules_resolved_count"),
        "lora_rank_matches": lora_adapter.get("r") == qlora_adapter.get("r"),
        "lora_alpha_matches": lora_adapter.get("alpha") == qlora_adapter.get("alpha"),
        "lora_dropout_matches": lora_adapter.get("dropout") == qlora_adapter.get("dropout"),
        "optimizer_matches": lora.get("training_detail", {}).get("optimizer")
        == qlora.get("training_detail", {}).get("optimizer"),
        "scheduler_matches": lora.get("training_detail", {}).get("lr_scheduler_type")
        == qlora.get("training_detail", {}).get("lr_scheduler_type"),
        "learning_rate_matches": lora.get("training_detail", {}).get("learning_rate")
        == qlora.get("training_detail", {}).get("learning_rate"),
        "effective_batch_matches": lora.get("training_detail", {}).get("effective_batch_size")
        == qlora.get("training_detail", {}).get("effective_batch_size"),
        "steps_match": lora.get("resources", {}).get("steps") == qlora.get("resources", {}).get("steps"),
        "train_examples_match": lora.get("dataset_training", {}).get("train_examples")
        == qlora.get("dataset_training", {}).get("train_examples"),
        "compute_dtype_matches": lora.get("quantization", {}).get("compute_dtype")
        == qlora.get("quantization", {}).get("compute_dtype"),
        "same_gpu": lora.get("resources", {}).get("gpu_name") == qlora.get("resources", {}).get("gpu_name"),
    }
    return {
        "checked": True,
        "all_matched": all(checks.values()),
        "checks": checks,
        "failed": [name for name, ok in checks.items() if not ok],
        "documented_differences": list(DOCUMENTED_DIFFERENCES),
    }


def build_analysis(variants: dict) -> dict:
    """Measured answers to the seven comparison questions."""
    base, lora, qlora = (variants.get(v, {}) for v in VARIANTS)
    bq, lq, qq = (v.get("quality", {}) for v in (base, lora, qlora))
    lr, qr = lora.get("resources", {}), qlora.get("resources", {})

    quality_deltas = {
        "lora_vs_base": {k: _delta(lq.get(k), bq.get(k)) for k, _, _ in metrics_mod.HEADLINE_METRICS},
        "qlora_vs_base": {k: _delta(qq.get(k), bq.get(k)) for k, _, _ in metrics_mod.HEADLINE_METRICS},
        "qlora_vs_lora": {k: _delta(qq.get(k), lq.get(k)) for k, _, _ in metrics_mod.HEADLINE_METRICS},
    }

    lora_vram = lr.get("peak_training_vram_bytes")
    qlora_vram = qr.get("peak_training_vram_bytes")
    vram = {
        "lora_peak_bytes": lora_vram,
        "qlora_peak_bytes": qlora_vram,
        "lora_peak_gib": lr.get("peak_training_vram_gib"),
        "qlora_peak_gib": qr.get("peak_training_vram_gib"),
        "qlora_saving_bytes": (lora_vram - qlora_vram) if None not in (lora_vram, qlora_vram) else None,
        "qlora_saving_gib": round((lora_vram - qlora_vram) / BYTES_PER_GIB, 4)
        if None not in (lora_vram, qlora_vram) else None,
        "qlora_saving_percent": (100.0 * (lora_vram - qlora_vram) / lora_vram)
        if lora_vram not in (None, 0) and qlora_vram is not None else None,
        "measured": None not in (lora_vram, qlora_vram),
    }

    lora_time = lr.get("training_duration_seconds")
    qlora_time = qr.get("training_duration_seconds")
    timing = {
        "lora_seconds": lora_time,
        "qlora_seconds": qlora_time,
        "lora_human": lr.get("training_duration_human"),
        "qlora_human": qr.get("training_duration_human"),
        "qlora_overhead_seconds": (qlora_time - lora_time) if None not in (lora_time, qlora_time) else None,
        "qlora_overhead_percent": (100.0 * (qlora_time - lora_time) / lora_time)
        if lora_time not in (None, 0) and qlora_time is not None else None,
        "qlora_relative_speed": (lora_time / qlora_time)
        if qlora_time not in (None, 0) and lora_time is not None else None,
        "lora_seconds_excluding_eval": lr.get("training_duration_excluding_evaluation"),
        "qlora_seconds_excluding_eval": qr.get("training_duration_excluding_evaluation"),
        "measured": None not in (lora_time, qlora_time),
    }

    lora_trainable = lora.get("training_parameters", {}).get("trainable") or lora.get(
        "parameters", {}
    ).get("trainable")
    qlora_trainable = qlora.get("training_parameters", {}).get("trainable") or qlora.get(
        "parameters", {}
    ).get("trainable")
    lora_total = lora.get("training_parameters", {}).get("total") or lora.get("parameters", {}).get("total")
    qlora_total = qlora.get("training_parameters", {}).get("total") or qlora.get("parameters", {}).get("total")
    parameters = {
        "base_total": base.get("parameters", {}).get("total"),
        "base_trainable": base.get("parameters", {}).get("trainable"),
        "lora_total": lora_total,
        "qlora_total": qlora_total,
        "lora_trainable": lora_trainable,
        "qlora_trainable": qlora_trainable,
        "trainable_identical": (lora_trainable == qlora_trainable)
        if None not in (lora_trainable, qlora_trainable) else None,
        "total_identical": (lora_total == qlora_total) if None not in (lora_total, qlora_total) else None,
        "lora_parameter_memory_gib": lora.get("training_parameters", {}).get("parameter_memory_gib"),
        "qlora_parameter_memory_gib": qlora.get("training_parameters", {}).get("parameter_memory_gib"),
        "note": "total parameter COUNT must match: quantization changes the memory footprint "
                "of the base weights, not how many parameters exist",
    }

    return {
        "quality_deltas": quality_deltas,
        "vram": vram,
        "training_time": timing,
        "parameters": parameters,
        "controlled_comparison": check_controlled_comparison(variants),
    }


def format_value(key: str, value) -> str:
    """Human-readable cell for the markdown table.

    ``None`` and NaN both mean "not measured" and render as ``N/A``. NaN shows
    up when a table column is round-tripped through pandas, which coerces None
    in a numeric column.
    """
    if value is None or (isinstance(value, float) and value != value):
        return "N/A"
    if key in ("total_parameters", "trainable_parameters"):
        return f"{int(value):,}"
    if key == "trainable_percent":
        return f"{value:.4f}%"
    if key == "training_duration_seconds":
        return f"{value:,.1f} ({format_duration(value)})"
    if key == "peak_training_vram_gib":
        return f"{value:.3f}"
    if key in ("eval_loss", "cer", "wer"):
        return f"{value:.4f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown_table(table: list[dict]) -> str:
    lines = ["| Metric | BASE | LoRA | QLoRA |", "|---|---:|---:|---:|"]
    for row in table:
        cells = " | ".join(format_value(row["key"], row[v]) for v in VARIANTS)
        lines.append(f"| {row['metric']} | {cells} |")
    return "\n".join(lines)


def render_csv(table: list[dict]) -> str:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["key", "metric", "group", "higher_is_better", *VARIANTS])
    for row in table:
        writer.writerow([
            row["key"], row["metric"], row["group"], row["higher_is_better"],
            *["" if row[v] is None else row[v] for v in VARIANTS],
        ])
    return buffer.getvalue()


def build_comparison(output_root: str) -> dict:
    """Everything the notebook needs, in one JSON-serializable payload."""
    variants = load_all(output_root)
    table = build_table(variants)
    return {
        "output_root": output_root,
        "variants": variants,
        "table": table,
        "analysis": build_analysis(variants),
        "markdown_table": render_markdown_table(table),
        "missing_artifacts": [
            path
            for record in variants.values()
            for name, path in record["paths"].items()
            if not os.path.isfile(path)
            and not (record["variant"] == "base" and name in ("resource_metrics", "training_log"))
        ],
    }


def load_loss_curves(output_root: str, variants: Iterable[str] = ("lora", "qlora")) -> dict:
    """Training and validation loss series per method, for the loss plots.

    BASE is absent on purpose: it is never trained, so it has no loss curve and
    must not be given a fabricated one.
    """
    curves = {}
    for variant in variants:
        path = artifact_paths(output_root, variant)["training_log"]
        if not os.path.isfile(path):
            continue
        history = read_json(path)
        curves[variant] = {
            "train": [
                {"step": e["step"], "epoch": e.get("epoch"), "loss": e["loss"]}
                for e in history if "loss" in e
            ],
            "eval": [
                {"step": e["step"], "epoch": e.get("epoch"), "loss": e["eval_loss"]}
                for e in history if "eval_loss" in e
            ],
        }
    return curves
