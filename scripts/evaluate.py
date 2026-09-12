#!/usr/bin/env python
"""Evaluate BASE, LoRA or QLoRA on the same held-out test set.

    python scripts/evaluate.py --model-variant base
    python scripts/evaluate.py --model-variant lora
    python scripts/evaluate.py --model-variant qlora

All three variants share the prompt, the test split, the image preprocessing,
the generation settings and the scoring code, so the only thing that varies is
the model under test.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from vlm_ft import data as data_mod  # noqa: E402
from vlm_ft import metrics as metrics_mod  # noqa: E402
from vlm_ft import modeling, resources  # noqa: E402
from vlm_ft.config import load_config  # noqa: E402
from vlm_ft.seeding import set_global_seed  # noqa: E402

LOGGER = logging.getLogger("evaluate")

VARIANTS = ("base", "lora", "qlora")
DEFAULT_CONFIG = {"base": "base.yaml", "lora": "lora.yaml", "qlora": "qlora.yaml"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-variant", required=True, choices=VARIANTS)
    parser.add_argument("--config", default=None, help="config file (default: configs/<variant>.yaml)")
    parser.add_argument("--adapter-dir", default=None,
                        help="trained adapter (default: <output_root>/<variant>/adapter)")
    parser.add_argument("--output-dir", default=None,
                        help="artifact directory (default: <output_root>/eval/<variant>)")
    parser.add_argument("--quantization", default="auto", choices=("auto", "none", "4bit"),
                        help="inference quantization. 'auto' reproduces the variant's training-time "
                             "setup: none for base/lora, 4-bit for qlora")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def resolve_quantization(variant: str, requested: str) -> bool:
    if requested == "auto":
        return variant == "qlora"
    return requested == "4bit"


def generate_predictions(model, processor, dataset, cfg, device):
    """Greedy-decode one prediction per test document."""
    import torch
    from torch.utils.data import DataLoader

    collator = data_mod.GenerationCollator(processor, cfg.prompt.instruction)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.evaluation.batch_size),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    gen_cfg = cfg.generation
    predictions: list[str] = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            prompt_len = batch["input_ids"].shape[1]
            output = model.generate(
                **batch,
                max_new_tokens=int(gen_cfg.max_new_tokens),
                do_sample=bool(gen_cfg.do_sample),
                num_beams=int(gen_cfg.num_beams),
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            for row in output:
                predictions.append(
                    processor.tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
                )
            if step % 10 == 0 or step == len(loader):
                LOGGER.info("generated %d/%d documents", len(predictions), len(dataset))
    return predictions


def compute_eval_loss(model, processor, dataset, cfg, device) -> dict:
    """Token-weighted teacher-forced cross-entropy over the test targets.

    Computed with byte-identical code for every variant. For QLoRA the forward
    pass runs against 4-bit base weights (unless --quantization overrides it),
    which is recorded in the artifact because it is a real, unavoidable
    difference in how the number is produced.
    """
    import torch
    from torch.utils.data import DataLoader

    collator = data_mod.JsonExtractionCollator(
        processor=processor,
        instruction=cfg.prompt.instruction,
        max_seq_length=int(cfg.data.max_seq_length),
        image_token_id=data_mod.resolve_image_token_id(processor, model),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.evaluation.batch_size),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    total_loss, total_tokens = 0.0, 0
    model.eval()
    # Teacher-forced scoring needs no KV cache; leaving it on would allocate one
    # per sequence for no benefit.
    previous_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    try:
        with torch.no_grad():
            for batch in loader:
                batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
                n_tokens = int((batch["labels"] != -100).sum())
                if n_tokens == 0:
                    continue
                loss = model(**batch).loss
                if loss is None or not torch.isfinite(loss):
                    continue
                total_loss += float(loss) * n_tokens
                total_tokens += n_tokens
    finally:
        if previous_use_cache is not None:
            model.config.use_cache = previous_use_cache
    return {
        "eval_loss": (total_loss / total_tokens) if total_tokens else None,
        "eval_loss_target_tokens": total_tokens,
        "eval_loss_definition": "token-weighted mean cross-entropy over target tokens only "
                                "(prompt and image tokens masked)",
        **collator.stats.as_dict(),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    import torch

    variant = args.model_variant
    config_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "configs", DEFAULT_CONFIG[variant]
    )
    cfg = load_config(config_path, args.overrides)

    use_4bit = resolve_quantization(variant, args.quantization)
    output_dir = args.output_dir or os.path.join(cfg.experiment.output_root, "eval", variant)
    os.makedirs(output_dir, exist_ok=True)

    adapter_dir = args.adapter_dir
    if variant != "base" and adapter_dir is None:
        adapter_dir = os.path.join(cfg.experiment.output_root, variant, "adapter")
    if variant != "base" and not os.path.isdir(adapter_dir):
        raise FileNotFoundError(
            f"adapter directory not found: {adapter_dir}\n"
            f"run: python scripts/train.py --method {variant}"
        )

    if variant != "base":
        # BASE loads no adapter, so the peft dispatch path is not exercised.
        modeling.run_preflight_checks()

    set_global_seed(int(cfg.experiment.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = modeling.resolve_compute_dtype(cfg.model.get("compute_dtype"))

    # A throwaway spec: evaluation only needs to know whether to quantize.
    spec = modeling.MethodSpec(
        name=variant, quantize_base=use_4bit, prepare_for_kbit=False,
        description=f"evaluation of {variant}",
    )

    splits = data_mod.load_splits(cfg, ("test",))
    test_set = splits["test"]
    processor = modeling.load_processor(cfg)

    resources.reset_peak_memory()
    with resources.Stopwatch("model_load") as load_timer:
        quantization_config = modeling.build_quantization_config(cfg, spec, compute_dtype) if use_4bit else None
        model = modeling.load_base_model(cfg, spec, compute_dtype, quantization_config)
        base_parameters = resources.count_parameters(model)
        if adapter_dir:
            model = modeling.load_adapter(model, adapter_dir)
            LOGGER.info("loaded adapter from %s", adapter_dir)
        if not torch.cuda.is_available():
            model = model.to(device)
        model.config.use_cache = True
    load_peak = resources.peak_memory()

    parameters = resources.count_parameters(model)
    adapter_params = resources.count_adapter_parameters(model)
    # BASE is not fine-tuned, so its trainable count for this experiment is 0.
    # For LoRA/QLoRA the adapter tensors on the instantiated model are the
    # parameters that were trained, even though they are frozen for inference.
    trained_params = 0 if variant == "base" else adapter_params
    parameters.update({
        "trainable": trained_params,
        "trainable_ratio": (trained_params / parameters["total"]) if parameters["total"] else 0.0,
        "trainable_percent": (100.0 * trained_params / parameters["total"]) if parameters["total"] else 0.0,
        "adapter_parameters": adapter_params,
        "base_model_total": base_parameters["total"],
        "trainable_source": (
            "0 by definition: BASE is evaluated without fine-tuning"
            if variant == "base"
            else "counted from the LoRA tensors of the instantiated evaluation model"
        ),
    })

    # ------------------------------------------------------------ generation
    resources.reset_peak_memory()
    with resources.Stopwatch("generation") as gen_timer:
        predictions = generate_predictions(model, processor, test_set, cfg, device)
    generation_peak = resources.peak_memory()

    document_scores, records = [], []
    for index, (prediction, example) in enumerate(zip(predictions, test_set)):
        reference = data_mod.extract_reference(example["ground_truth"])
        score = metrics_mod.score_document(prediction, reference)
        document_scores.append(score)
        records.append({
            "index": index,
            "json_valid": score["json_valid"],
            "exact_match": score["exact_match"],
            "matched_fields": score["matched"],
            "reference_fields": score["reference_fields"],
            "predicted_fields": score["predicted_fields"],
            "char_distance": score["char_distance"],
            "char_reference_length": score["char_reference_length"],
            "prediction_raw": prediction,
            "prediction_canonical": score["prediction_text_canonical"],
            "reference": score["reference_text"],
        })

    aggregate = metrics_mod.aggregate(document_scores)

    # ------------------------------------------------------------ eval loss
    loss_report = {"eval_loss": None, "eval_loss_skipped": "evaluation.compute_eval_loss is false"}
    if bool(cfg.evaluation.get("compute_eval_loss", True)):
        with resources.Stopwatch("eval_loss") as loss_timer:
            loss_report = compute_eval_loss(model, processor, test_set, cfg, device)
        loss_report["duration_seconds"] = round(loss_timer.duration_seconds or 0.0, 4)
    aggregate["eval_loss"] = loss_report.get("eval_loss")

    payload = {
        "variant": variant,
        "schema_version": 1,
        "metrics": aggregate,
        "eval_loss_detail": loss_report,
        "parameters": parameters,
        "inference": {
            "quantized_4bit": use_4bit,
            "quantization_flag": args.quantization,
            "compute_dtype": modeling.dtype_name(compute_dtype),
            "device": str(device),
            "generation": {
                "max_new_tokens": int(cfg.generation.max_new_tokens),
                "do_sample": bool(cfg.generation.do_sample),
                "num_beams": int(cfg.generation.num_beams),
            },
            "model_load": {**load_timer.as_dict(), **load_peak},
            "generation_timing": {**gen_timer.as_dict(), **generation_peak},
        },
        "dataset": {
            "id": cfg.data.dataset_id,
            "split": cfg.data.test_split,
            "examples": len(test_set),
        },
        "model": {
            "id": cfg.model.id,
            "revision": modeling.resolve_revision(cfg),
            "adapter_dir": adapter_dir,
        },
        "gpu": resources.gpu_info(),
    }

    resources.write_json(os.path.join(output_dir, "metrics.json"), payload)
    with open(os.path.join(output_dir, "predictions.jsonl"), "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    LOGGER.info("wrote artifacts to %s", output_dir)

    print(json.dumps({
        "variant": variant,
        **{key: round(aggregate[key], 4) if isinstance(aggregate.get(key), float) else aggregate.get(key)
           for key, _, _ in metrics_mod.HEADLINE_METRICS},
        "trainable_parameters": parameters["trainable"],
        "total_parameters": parameters["total"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
