#!/usr/bin/env python
"""Run the whole comparison across several seeds.

    python scripts/run_experiment.py --seeds 42 43 44

For each seed this drives the same CLI a single run would use, writing into
<output-root>/seed<N>/, then aggregates across seeds. Nothing about the training
or evaluation logic lives here -- this is orchestration only, so a multi-seed
study and a single run go through byte-identical code.

Both methods use the same seed list, which makes the observations paired: the
per-seed LoRA-minus-QLoRA difference is the unit of comparison.

Re-running skips work that already produced its artifact, so an interrupted
study (a reclaimed Colab session, say) resumes where it stopped. Pass
--force to redo everything.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from vlm_ft import report, resources  # noqa: E402
from vlm_ft.config import load_config  # noqa: E402

LOGGER = logging.getLogger("run_experiment")
SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                        help="seeds to run (default: 42 43 44)")
    parser.add_argument("--methods", nargs="+", default=["lora", "qlora"],
                        choices=["lora", "qlora"])
    parser.add_argument("--variants", nargs="+", default=["base", "lora", "qlora"],
                        choices=["base", "lora", "qlora"], help="variants to evaluate")
    parser.add_argument("--output-root", default=None,
                        help="default: experiment.output_root from configs/base.yaml")
    parser.add_argument("--force", action="store_true", help="re-run steps whose artifact exists")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run to validate the pipeline (4 steps, 8 train / 4 test examples)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="config override passed to every step")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


SMOKE_OVERRIDES = [
    "training.max_steps=4",
    "data.max_train_samples=8",
    "data.max_eval_samples=4",
    "data.max_test_samples=4",
    "generation.max_new_tokens=64",
]


def build_command(kind: str, name: str, seed: int, root: str, overrides: list[str]) -> list[str]:
    script = "train.py" if kind == "train" else "evaluate.py"
    flag = "--method" if kind == "train" else "--model-variant"
    command = [sys.executable, os.path.join(SCRIPTS, script), flag, name]
    for override in [f"experiment.seed={seed}", f"experiment.output_root={root}", *overrides]:
        command += ["--set", override]
    return command


def artifact_for(kind: str, name: str, root: str) -> str:
    if kind == "train":
        return os.path.join(root, name, "resource_metrics.json")
    return os.path.join(root, "eval", name, "metrics.json")


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    base_config = os.path.join(SCRIPTS, "..", "configs", "base.yaml")
    output_root = args.output_root or load_config(base_config).experiment.output_root
    overrides = list(args.overrides) + (SMOKE_OVERRIDES if args.smoke else [])

    steps: list[tuple[str, str, int]] = []
    for seed in args.seeds:
        steps += [("train", method, seed) for method in args.methods]
        steps += [("evaluate", variant, seed) for variant in args.variants]

    LOGGER.info(
        "seeds=%s methods=%s variants=%s -> %d steps under %s",
        args.seeds, args.methods, args.variants, len(steps), output_root,
    )
    if args.smoke:
        LOGGER.warning("SMOKE MODE: results are for pipeline validation only, not for analysis")

    completed, skipped, failed = [], [], []
    started = time.perf_counter()

    for index, (kind, name, seed) in enumerate(steps, start=1):
        root = report.seed_root(output_root, seed)
        artifact = artifact_for(kind, name, root)
        label = f"seed {seed} | {kind} {name}"

        if not args.force and os.path.isfile(artifact):
            LOGGER.info("[%d/%d] %s -- already done, skipping (%s)", index, len(steps), label, artifact)
            skipped.append(label)
            continue

        command = build_command(kind, name, seed, root, overrides)
        LOGGER.info("[%d/%d] %s", index, len(steps), label)
        print("+", " ".join(command), flush=True)
        if args.dry_run:
            continue

        step_started = time.perf_counter()
        result = subprocess.run(command)
        elapsed = time.perf_counter() - step_started
        if result.returncode != 0:
            LOGGER.error("%s FAILED with exit code %d after %s",
                         label, result.returncode, resources.format_duration(elapsed))
            failed.append(label)
            # Keep going: a later seed may still succeed, and the aggregation
            # reports whatever is present rather than guessing.
            continue
        LOGGER.info("%s done in %s", label, resources.format_duration(elapsed))
        completed.append(label)

    if args.dry_run:
        print(f"\ndry run: {len(steps)} steps would run")
        return 0

    # ---------------------------------------------------------- aggregation
    for seed in args.seeds:
        root = report.seed_root(output_root, seed)
        if os.path.isdir(root):
            subprocess.run(
                [sys.executable, os.path.join(SCRIPTS, "collect_results.py"),
                 "--output-root", root, "--quiet"],
                check=False,
            )

    collect = [sys.executable, os.path.join(SCRIPTS, "collect_results.py"),
               "--output-root", output_root, "--seeds", *[str(s) for s in args.seeds]]
    subprocess.run(collect, check=False)

    total = time.perf_counter() - started
    print("\n" + "=" * 78)
    print(f"completed {len(completed)} | skipped {len(skipped)} | failed {len(failed)}")
    print(f"total wall clock: {resources.format_duration(total)}")
    if failed:
        print("\nFAILED steps (their metrics are reported as N/A, never estimated):")
        for label in failed:
            print("  -", label)
    print(f"\nmulti-seed artifacts: {os.path.join(output_root, 'multiseed')}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
