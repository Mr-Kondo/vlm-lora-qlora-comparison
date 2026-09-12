#!/usr/bin/env python
"""Single training entry point for both fine-tuning methods.

    python scripts/train.py --method lora  --config configs/lora.yaml
    python scripts/train.py --method qlora --config configs/qlora.yaml

Dataset loading, preprocessing, target formatting, the training loop,
checkpointing, logging, metric recording, seeding and the artifact layout are
shared. The method only decides how the base weights are loaded and whether the
model is prepared for k-bit training; see vlm_ft.modeling.METHOD_SPECS.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from vlm_ft import data as data_mod  # noqa: E402
from vlm_ft import modeling, resources  # noqa: E402
from vlm_ft.config import config_fingerprint, load_config  # noqa: E402
from vlm_ft.seeding import set_global_seed  # noqa: E402

LOGGER = logging.getLogger("train")

#: Config keys that are *allowed* to differ between the two methods. Every other
#: key contributes to the fingerprint that proves the runs were matched.
METHOD_SPECIFIC_KEYS = ("method", "quantization", "_config_path", "_parent_config")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=sorted(modeling.METHOD_SPECS),
                        help="fine-tuning method to run")
    parser.add_argument("--config", default=None,
                        help="config file (default: configs/<method>.yaml)")
    parser.add_argument("--output-dir", default=None,
                        help="artifact directory (default: <output_root>/<method>)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set training.num_train_epochs=1")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def compute_total_steps(cfg, num_train_examples: int) -> int:
    """Total optimizer steps the run will take, used to derive warmup length."""
    import math

    max_steps = int(cfg.training.get("max_steps", -1))
    if max_steps > 0:
        return max_steps
    effective_batch = int(cfg.training.per_device_train_batch_size) * int(
        cfg.training.gradient_accumulation_steps
    )
    steps_per_epoch = max(1, math.ceil(num_train_examples / effective_batch))
    return max(1, int(math.ceil(steps_per_epoch * float(cfg.training.num_train_epochs))))


def resolve_warmup_steps(cfg, total_steps: int) -> int:
    """Warmup length in steps.

    transformers 5 dropped ``warmup_ratio`` from TrainingArguments, so the ratio
    is converted here instead. Doing the conversion ourselves keeps the schedule
    identical for LoRA and QLoRA and independent of the transformers version.
    An explicit ``training.warmup_steps`` takes precedence when set.
    """
    import math

    explicit = int(cfg.training.get("warmup_steps", 0) or 0)
    if explicit > 0:
        return explicit
    ratio = float(cfg.training.get("warmup_ratio", 0.0) or 0.0)
    return int(math.ceil(total_steps * ratio))


def build_training_arguments(cfg, output_dir: str, compute_dtype, num_train_examples: int):
    import torch
    from transformers import TrainingArguments

    tcfg = cfg.training
    total_steps = compute_total_steps(cfg, num_train_examples)
    warmup_steps = resolve_warmup_steps(cfg, total_steps)
    LOGGER.info(
        "schedule: ~%d optimizer steps, %d warmup steps (%s)",
        total_steps, warmup_steps, tcfg.lr_scheduler_type,
    )
    return TrainingArguments(
        output_dir=os.path.join(output_dir, "trainer"),
        num_train_epochs=float(tcfg.num_train_epochs),
        max_steps=int(tcfg.get("max_steps", -1)),
        per_device_train_batch_size=int(tcfg.per_device_train_batch_size),
        gradient_accumulation_steps=int(tcfg.gradient_accumulation_steps),
        per_device_eval_batch_size=int(tcfg.per_device_eval_batch_size),
        learning_rate=float(tcfg.learning_rate),
        lr_scheduler_type=str(tcfg.lr_scheduler_type),
        warmup_steps=warmup_steps,
        weight_decay=float(tcfg.weight_decay),
        max_grad_norm=float(tcfg.max_grad_norm),
        optim=str(tcfg.optim),
        logging_steps=int(tcfg.logging_steps),
        eval_strategy=str(tcfg.eval_strategy),
        eval_steps=int(tcfg.eval_steps),
        save_strategy=str(tcfg.save_strategy),
        gradient_checkpointing=bool(tcfg.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=int(tcfg.dataloader_num_workers),
        report_to=list(tcfg.get("report_to") or []),
        remove_unused_columns=False,
        label_names=["labels"],
        seed=int(cfg.experiment.seed),
        data_seed=int(cfg.experiment.seed),
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        logging_first_step=True,
        disable_tqdm=False,
    )


def cast_trainable_to_fp32(model) -> int:
    """Keep adapter weights in fp32 under mixed precision.

    Applied identically to LoRA and QLoRA. Without it, fp16 training raises
    "Attempting to unscale FP16 gradients" because the adapters inherit the
    base layer dtype. The memory cost is negligible (adapters are <1% of the
    model) and keeping it symmetric avoids introducing a method-dependent
    numerical difference.
    """
    import torch

    converted = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype in (torch.float16, torch.bfloat16):
            param.data = param.data.to(torch.float32)
            converted += 1
    return converted


def summarize_log_history(log_history: list[dict]) -> dict:
    """Pull headline numbers and the in-training evaluation cost out of the log."""
    train_losses = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    eval_losses = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e]
    eval_seconds = sum(e.get("eval_runtime", 0.0) for e in log_history if "eval_runtime" in e)
    return {
        "final_train_loss": train_losses[-1][1] if train_losses else None,
        "first_train_loss": train_losses[0][1] if train_losses else None,
        "final_eval_loss": eval_losses[-1][1] if eval_losses else None,
        "best_eval_loss": min((v for _, v in eval_losses), default=None),
        "best_eval_loss_step": min(eval_losses, key=lambda kv: kv[1])[0] if eval_losses else None,
        "num_train_log_points": len(train_losses),
        "num_eval_log_points": len(eval_losses),
        "evaluation_seconds_within_training": round(eval_seconds, 4),
    }


def environment_report() -> dict:
    import torch
    import transformers

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
    }
    for name in ("peft", "datasets", "accelerate", "bitsandbytes"):
        try:
            report[name] = __import__(name).__version__
        except Exception as exc:
            report[name] = f"unavailable: {type(exc).__name__}"
    return report


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    config_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "configs", f"{args.method}.yaml"
    )
    cfg = load_config(config_path, args.overrides)
    spec = modeling.validate_method_against_config(args.method, cfg)

    output_dir = args.output_dir or os.path.join(cfg.experiment.output_root, args.method)
    os.makedirs(output_dir, exist_ok=True)

    LOGGER.info("method=%s (%s)", spec.name, spec.description)
    LOGGER.info("config=%s output=%s", config_path, output_dir)

    set_global_seed(int(cfg.experiment.seed))

    gpu = resources.gpu_info()
    if not gpu["available"]:
        LOGGER.warning("no CUDA device: %s", gpu["unavailable_reason"])

    compute_dtype = modeling.resolve_compute_dtype(cfg.model.get("compute_dtype"))
    LOGGER.info("compute dtype=%s", modeling.dtype_name(compute_dtype))

    # ---------------------------------------------------------------- data
    splits = data_mod.load_splits(cfg, ("train", "validation"))
    processor = modeling.load_processor(cfg)

    # ------------------------------------------------- model preparation
    # Timed and memory-profiled separately so quantization overhead never leaks
    # into the reported training duration.
    resources.reset_peak_memory()
    with resources.Stopwatch("model_preparation") as prep_timer:
        quantization_config = modeling.build_quantization_config(cfg, spec, compute_dtype)
        model = modeling.load_base_model(cfg, spec, compute_dtype, quantization_config)
        base_params = resources.count_parameters(model)
        target_modules = modeling.resolve_target_modules(
            model, cfg.lora.target_modules_suffixes, cfg.lora.target_scope
        )
        model = modeling.prepare_base_for_training(
            model, spec, bool(cfg.training.gradient_checkpointing)
        )
        model = modeling.attach_lora(model, cfg, target_modules)
        cast_trainable_to_fp32(model)
    prep_peak = resources.peak_memory()

    parameters = resources.count_parameters(model)
    parameters["base_model_total"] = base_params["total"]
    LOGGER.info(
        "parameters: total=%s trainable=%s (%.4f%%) | targeted modules=%d",
        f"{parameters['total']:,}",
        f"{parameters['trainable']:,}",
        parameters["trainable_percent"],
        len(target_modules),
    )

    collator = data_mod.JsonExtractionCollator(
        processor=processor,
        instruction=cfg.prompt.instruction,
        max_seq_length=int(cfg.data.max_seq_length),
        image_token_id=data_mod.resolve_image_token_id(processor, model),
    )

    from transformers import Trainer

    training_args = build_training_arguments(
        cfg, output_dir, compute_dtype, len(splits["train"])
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        data_collator=collator,
        processing_class=processor,
    )

    # ------------------------------------------------------------ training
    resources.reset_peak_memory()
    with resources.Stopwatch("training") as train_timer:
        train_result = trainer.train()
    train_peak = resources.peak_memory()

    log_history = list(trainer.state.log_history)
    log_summary = summarize_log_history(log_history)
    duration = train_timer.duration_seconds or 0.0
    eval_seconds = log_summary["evaluation_seconds_within_training"]

    # ----------------------------------------------------------- artifacts
    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    LOGGER.info("saved adapter to %s", adapter_dir)

    steps = int(trainer.state.global_step)
    effective_batch = int(cfg.training.per_device_train_batch_size) * int(
        cfg.training.gradient_accumulation_steps
    )
    samples = steps * effective_batch

    resource_metrics = resources.build_resource_metrics(
        method=spec.name,
        gpu=gpu,
        training_peak=train_peak,
        training_timer=train_timer,
        parameters=parameters,
        steps=steps,
        samples=samples,
        extra={
            "model_preparation": {
                **prep_timer.as_dict(),
                **prep_peak,
                "note": "model download/load, quantization and adapter attachment; "
                        "excluded from training duration on purpose",
            },
            "quantization": {
                "enabled": spec.quantize_base,
                "quant_type": cfg.quantization.get("bnb_4bit_quant_type") if spec.quantize_base else None,
                "double_quant": cfg.quantization.get("bnb_4bit_use_double_quant") if spec.quantize_base else None,
                "compute_dtype": modeling.dtype_name(compute_dtype),
                "prepared_for_kbit_training": spec.prepare_for_kbit,
            },
            "adapter": modeling.adapter_summary(cfg, target_modules),
            "training_detail": {
                "epochs_requested": float(cfg.training.num_train_epochs),
                "epochs_completed": float(train_result.metrics.get("epoch", 0.0)),
                "effective_batch_size": effective_batch,
                "optimizer": str(cfg.training.optim),
                "learning_rate": float(cfg.training.learning_rate),
                "lr_scheduler_type": str(cfg.training.lr_scheduler_type),
                "warmup_steps": int(training_args.warmup_steps),
                "mixed_precision": modeling.dtype_name(compute_dtype),
                "gradient_checkpointing": bool(cfg.training.gradient_checkpointing),
                "duration_seconds_excluding_evaluation": round(max(duration - eval_seconds, 0.0), 4),
                "duration_human_excluding_evaluation": resources.format_duration(
                    max(duration - eval_seconds, 0.0)
                ),
                "trainer_reported_runtime_seconds": train_result.metrics.get("train_runtime"),
                **log_summary,
            },
            "dataset": {
                "id": cfg.data.dataset_id,
                "train_examples": len(splits["train"]),
                "validation_examples": len(splits["validation"]),
                **collator.stats.as_dict(),
            },
            "model": {
                "id": cfg.model.id,
                "revision": modeling.resolve_revision(cfg),
                "source": modeling.model_source(cfg),
            },
            "config_fingerprint_excluding_method_keys": config_fingerprint(cfg, METHOD_SPECIFIC_KEYS),
            "environment": environment_report(),
        },
    )
    resources.write_json(os.path.join(output_dir, "resource_metrics.json"), resource_metrics)
    resources.write_json(os.path.join(output_dir, "training_log_history.json"), log_history)
    resources.write_json(
        os.path.join(output_dir, "run_config.json"),
        {"method": spec.name, "config_path": config_path, "overrides": args.overrides,
         "resolved_config": cfg.to_dict()},
    )

    print(json.dumps({
        "method": spec.name,
        "training_duration": resource_metrics["training"]["duration_human"],
        "peak_vram_gib": resource_metrics["gpu"]["peak_memory_allocated_gib"],
        "trainable_parameters": parameters["trainable"],
        "trainable_percent": round(parameters["trainable_percent"], 4),
        "final_train_loss": log_summary["final_train_loss"],
        "best_eval_loss": log_summary["best_eval_loss"],
        "artifacts": output_dir,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
