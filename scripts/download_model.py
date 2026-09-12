#!/usr/bin/env python
"""Pre-fetch the base model (and optionally the dataset) before training.

    python scripts/download_model.py
    python scripts/download_model.py --local-dir models/smolvlm --skip-dataset

Downloading up front keeps model transfer time out of the measured training
duration and pins the run to one resolved commit SHA, which is recorded in
outputs/download_manifest.json and echoed into every run artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from vlm_ft import resources  # noqa: E402
from vlm_ft.config import load_config  # noqa: E402

LOGGER = logging.getLogger("download")

#: Large artifacts in the repo that this experiment never loads. SmolVLM-Instruct
#: ships ~24 GB of ONNX exports next to a 4.5 GB safetensors checkpoint.
DEFAULT_IGNORE = ["onnx/*", "*.onnx", "*.onnx_data", "*.png", "*.jpg", "*.pdf", "*.msgpack", "*.h5"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="config file (default: configs/base.yaml)")
    parser.add_argument("--local-dir", default=None,
                        help="download into this directory instead of the shared HF cache")
    parser.add_argument("--skip-dataset", action="store_true", help="do not pre-fetch the dataset")
    parser.add_argument("--skip-model", action="store_true", help="do not pre-fetch the model")
    parser.add_argument("--keep-onnx", action="store_true",
                        help="also download ONNX exports (roughly 24 GB for SmolVLM)")
    parser.add_argument("--output", default=None,
                        help="manifest path (default: <output_root>/download_manifest.json)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    config_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "configs", "base.yaml"
    )
    cfg = load_config(config_path, args.overrides)
    manifest: dict = {"config_path": config_path, "model": None, "dataset": None}

    if not args.skip_model:
        from huggingface_hub import model_info, snapshot_download

        requested = cfg.model.get("revision") or "main"
        info = model_info(cfg.model.id, revision=requested)
        LOGGER.info("resolved %s@%s -> %s", cfg.model.id, requested, info.sha)

        path = snapshot_download(
            repo_id=cfg.model.id,
            revision=info.sha,
            local_dir=args.local_dir,
            ignore_patterns=None if args.keep_onnx else DEFAULT_IGNORE,
        )
        size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(path)
            for f in files
            if not os.path.islink(os.path.join(root, f))
        )
        manifest["model"] = {
            "id": cfg.model.id,
            "requested_revision": requested,
            "resolved_sha": info.sha,
            "path": path,
            "bytes_on_disk": size,
            "gib_on_disk": round(size / resources.BYTES_PER_GIB, 3),
            "ignored_patterns": None if args.keep_onnx else DEFAULT_IGNORE,
        }
        LOGGER.info("model ready at %s (%.2f GiB)", path, size / resources.BYTES_PER_GIB)

    if not args.skip_dataset:
        from datasets import load_dataset

        splits = {}
        for key in ("train_split", "validation_split", "test_split"):
            name = cfg.data[key]
            dataset = load_dataset(
                cfg.data.dataset_id, split=name, revision=cfg.data.get("dataset_revision")
            )
            splits[name] = len(dataset)
            LOGGER.info("dataset split %s: %d examples", name, len(dataset))
        manifest["dataset"] = {"id": cfg.data.dataset_id, "splits": splits}

    output = args.output or os.path.join(cfg.experiment.output_root, "download_manifest.json")
    resources.write_json(output, manifest)

    if manifest["model"] and args.local_dir:
        LOGGER.info(
            "pass this to training so it loads from disk:\n"
            "    --set model.local_dir=%s", manifest["model"]["path"],
        )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
