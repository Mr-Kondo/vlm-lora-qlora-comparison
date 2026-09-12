#!/usr/bin/env python
"""Merge the run artifacts into the final comparison table.

    python scripts/collect_results.py

Writes comparison.json, comparison.csv and comparison.md under
<output_root>/comparison/. Every value is read from an artifact; anything that
was not measured is reported as N/A rather than filled in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from vlm_ft import report, resources  # noqa: E402
from vlm_ft.config import load_config  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default=None, help="default: experiment.output_root from the config")
    parser.add_argument("--config", default=None, help="config file (default: configs/base.yaml)")
    parser.add_argument("--output-dir", default=None, help="default: <output_root>/comparison")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="also aggregate across <output-root>/seed<N>/ directories")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _collect_multi_seed(args, output_root: str) -> int:
    """Aggregate <output-root>/seed<N>/ into one multi-seed summary."""
    comparison = report.build_multi_seed_comparison(output_root, args.seeds)
    output_dir = args.output_dir or os.path.join(output_root, "multiseed")
    os.makedirs(output_dir, exist_ok=True)

    resources.write_json(os.path.join(output_dir, "multiseed_comparison.json"), comparison)
    with open(os.path.join(output_dir, "multiseed_summary.csv"), "w", encoding="utf-8") as handle:
        handle.write(report.render_multi_seed_csv(comparison["table"], comparison["seeds_found"]))
    with open(os.path.join(output_dir, "multiseed_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("# BASE vs LoRA vs QLoRA across seeds\n\n" + comparison["markdown_table"] + "\n")
    resources.write_json(
        os.path.join(output_dir, "multiseed_loss_curves.json"),
        report.load_multi_seed_loss_curves(output_root, comparison["seeds_found"]),
    )

    if not args.quiet:
        print(comparison["markdown_table"])
        if comparison["seeds_missing"]:
            print(f"\nseeds requested but not found: {comparison['seeds_missing']}")

        paired = comparison["paired_lora_vs_qlora"]
        print("\nPaired LoRA vs QLoRA (per-seed differences, same seeds for both methods):")
        for key, entry in paired["metrics"].items():
            diff = entry["difference"]
            if diff["n"] == 0:
                print(f"  {entry['label']:<22} not measured")
                continue
            spread = "" if diff["std"] is None else f" ± {diff['std']:.4f}"
            print(f"  {entry['label']:<22} mean diff {diff['mean']:+.4f}{spread}"
                  f"  (QLoRA better in {entry['seeds_favouring_candidate']}/{diff['n']} seeds"
                  f"{', consistent' if entry['sign_is_consistent'] else ''})")

        check = comparison["controlled_comparison"]
        if check["all_seeds_matched"] is None:
            print("\nControlled comparison: not checked "
                  "(needs both a LoRA and a QLoRA training artifact)")
        else:
            print(f"\nControlled comparison: seeds checked {check['seeds_checked']}, "
                  f"all matched: {check['all_seeds_matched']}")
        if check["failed_seeds"]:
            print(f"  MISMATCH in seeds {check['failed_seeds']} -- fix before concluding")
        if check["gpu_warning"]:
            print(f"  WARNING: {check['gpu_warning']}")

    print(f"\nwrote {output_dir}/multiseed_summary.{{json,csv,md}}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    output_root = args.output_root
    if output_root is None:
        config_path = args.config or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "configs", "base.yaml"
        )
        output_root = load_config(config_path).experiment.output_root

    if args.seeds:
        return _collect_multi_seed(args, output_root)

    comparison = report.build_comparison(output_root)
    output_dir = args.output_dir or os.path.join(output_root, "comparison")
    os.makedirs(output_dir, exist_ok=True)

    resources.write_json(os.path.join(output_dir, "comparison.json"), comparison)
    with open(os.path.join(output_dir, "comparison.csv"), "w", encoding="utf-8") as handle:
        handle.write(report.render_csv(comparison["table"]))
    with open(os.path.join(output_dir, "comparison.md"), "w", encoding="utf-8") as handle:
        handle.write("# BASE vs LoRA vs QLoRA\n\n" + comparison["markdown_table"] + "\n")
    resources.write_json(
        os.path.join(output_dir, "loss_curves.json"), report.load_loss_curves(output_root)
    )

    if not args.quiet:
        print(comparison["markdown_table"])
        if comparison["missing_artifacts"]:
            print("\nMissing artifacts (reported as N/A above):")
            for path in comparison["missing_artifacts"]:
                print(f"  - {path}")
        check = comparison["analysis"]["controlled_comparison"]
        if check.get("checked"):
            status = "MATCHED" if check["all_matched"] else f"MISMATCH: {check['failed']}"
            print(f"\nControlled-comparison check: {status}")
    print(f"\nwrote {output_dir}/comparison.{{json,csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
