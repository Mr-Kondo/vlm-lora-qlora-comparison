# BASE vs LoRA vs QLoRA — document-image → JSON

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mr-Kondo/vlm-lora-qlora-comparison/blob/main/notebooks/vlm_lora_qlora_comparison.ipynb)

*日本語版: [README.ja.md](README.ja.md)*

A controlled comparison of three conditions on a vision-language model fine-tuned to convert receipt
photographs into structured JSON:

| Condition | Base weights | Adapter | Trainable parameters |
|---|---|---|---|
| **BASE** | 16-bit, unmodified | none | 0 |
| **LoRA** | 16-bit | LoRA | adapter only |
| **QLoRA** | 4-bit NF4 (double quant) | equivalent LoRA | adapter only |

The experiment is not only about quality. It measures four axes together —

**generation quality × GPU VRAM × training time × trainable parameters**

— so the quality/resource trade-off between the three conditions is explicit rather than implied.

- **Model:** [`HuggingFaceTB/SmolVLM-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) (2.2 B, Idefics3)
- **Dataset:** [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2) — 800 train / 100 validation / 100 test receipts
- **Task:** image → canonical JSON (`menu`, `sub_total`, `total`, `void_menu`)

---

## Quick start

### Google Colab

Click the badge above. It opens `notebooks/vlm_lora_qlora_comparison.ipynb` in Colab; the first
setup cell clones this repository into `/content/vlm-lora-qlora-comparison` on its own, so there is
nothing to upload. Then:

1. **Runtime → Change runtime type → T4 GPU** (or better). Without a GPU neither training nor VRAM
   measurement can run, and the first cell says so.
2. Run the cells top to bottom. The notebook drives the CLI below and then reads the artifacts to
   build the tables and plots.

`torch` and `torchvision` are deliberately absent from `requirements.txt` — Colab ships them matched
to the runtime's CUDA build. If imports complain after the install cell, use
**Runtime → Restart session**, then re-run from the version-check cell (skip the install).

**Run a smoke test before the real thing.** Append this to a training cell to exercise the whole
flow in a couple of minutes, then remove it:

```
--set training.max_steps=4 --set data.max_train_samples=8 --set data.max_eval_samples=4
```

**Expected duration** with the default config (800 examples × 2 epochs): roughly 1–2 hours for both
training runs combined on an L4/A100, and appreciably longer on a T4 — plan for 3–4 hours there.

**Train both methods in the same session.** Colab may hand a different GPU to a new session, which
would make the VRAM and training-time comparison meaningless. If you do have to split the work,
check the `same_gpu` line in the controlled-comparison output (notebook section 7): when it fails,
the quality comparison still holds but the resource comparison does not.

**To survive a disconnect**, point the output root at Drive before training — one edit covers
training, evaluation and the notebook's own reads:

```python
from google.colab import drive; drive.mount('/content/drive')
!sed -i 's|output_root: outputs|output_root: /content/drive/MyDrive/vlm_ft_outputs|' configs/base.yaml
```

Optionally cache the ~7 GB of model and dataset downloads there too, at the cost of slower loads
(set this before the first `!python` cell so subprocesses inherit it):

```python
import os; os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
```

### Local / any CUDA host

```bash
pip install -r requirements.txt

python scripts/download_model.py

python scripts/train.py --method lora  --config configs/lora.yaml
python scripts/train.py --method qlora --config configs/qlora.yaml

python scripts/evaluate.py --model-variant base  --config configs/base.yaml
python scripts/evaluate.py --model-variant lora  --config configs/lora.yaml
python scripts/evaluate.py --model-variant qlora --config configs/qlora.yaml

python scripts/collect_results.py
```

### Multi-seed study (recommended)

A single run shows whether fine-tuning helps; it cannot tell you whether a small LoRA-vs-QLoRA gap
is real. `scripts/run_experiment.py` runs the whole matrix across several seeds and compares the
methods pairwise — both methods use the same seeds, so each seed yields one paired difference.

```bash
python scripts/run_experiment.py --seeds 42 43 44 --dry-run   # print the 15 steps
python scripts/run_experiment.py --seeds 42 43 44 --smoke     # validate the plumbing, minutes
python scripts/run_experiment.py --seeds 42 43 44             # the real study
```

It is orchestration only: it shells out to the same `train.py` and `evaluate.py` as above, writing
into `<output_root>/seed<N>/`. A step whose artifact already exists is skipped, so an interrupted
study (a reclaimed Colab session, say) resumes where it stopped; `--force` redoes everything. A
failed step does not abort the rest — it is listed at the end and reported as N/A.

Results land in `<output_root>/multiseed/` as mean ± sample standard deviation, plus the per-seed
values. Sign agreement is reported instead of a p-value: with a handful of seeds, "QLoRA is behind
in 3 of 3 seeds" is interpretable, while a significance test on three points is not.

Rehearse the whole flow in a couple of minutes before committing to a real run:

```bash
python scripts/train.py --method lora --set training.max_steps=4 --set data.max_train_samples=8
```

---

## One training entry point

There is no `train_lora.py` / `train_qlora.py`. `scripts/train.py --method {lora,qlora}` shares the
dataset loading, preprocessing, target formatting, trainer loop, checkpointing, logging, metric
recording, seeding and artifact layout. The method decides two things and nothing else
(`src/vlm_ft/modeling.py`, `METHOD_SPECS`):

```python
lora  -> quantize_base=False, prepare_for_kbit=False   # base held in the 16-bit compute dtype
qlora -> quantize_base=True,  prepare_for_kbit=True    # same base, 4-bit NF4, prepared for k-bit training
```

`configs/lora.yaml` and `configs/qlora.yaml` both inherit `configs/base.yaml` and differ only in the
method name and the quantization block — verifiable with `diff configs/lora.yaml configs/qlora.yaml`
and enforced by `tests/test_config_and_methods.py`.

Held identical: model id and revision, dataset and all three splits, prompt, target JSON format,
seed, epochs, effective batch size, LoRA rank / alpha / dropout / target modules, optimizer, LR
schedule and warmup, evaluation schedule, generation settings.

Two guards stop an accidental miscomparison:

- `--method lora` with `configs/qlora.yaml` (or a mismatched `load_in_4bit`) is a hard error.
- Each run records a fingerprint of its config with the method-specific keys removed;
  `collect_results.py` reports whether the two fingerprints match.

### Differences that cannot be removed

| Parameter | LoRA | QLoRA | Why |
|---|---|---|---|
| Base weight storage | 16-bit | 4-bit NF4 + double quant | This is the independent variable |
| `prepare_model_for_kbit_training` | not applied | applied | Required for a quantized base; meaningless otherwise |
| Evaluation weight precision | 16-bit | 4-bit (default) | QLoRA's deployable form; override with `--quantization none` |
| Evaluation loss arithmetic | 16-bit forward | 4-bit forward | Identical code, different precision |

The same list, with reasoning, is printed in the notebook and stored in every comparison artifact.

---

## What is measured

### Quality — `outputs/eval/<variant>/metrics.json`

Scored by `src/vlm_ft/metrics.py`, which is pure Python and unit-tested independently of any GPU run.
Both sides of every comparison are canonicalized the same way: single-entry `menu` objects become
one-element lists, keys are sorted, values are whitespace-normalized strings.

| Metric | Definition |
|---|---|
| JSON validity rate | Fraction of generations that parse to a JSON object (markdown fences and surrounding prose are stripped first, identically for all variants) |
| Document exact match | Canonical predicted document equals the canonical reference |
| Field accuracy | Fraction of ground-truth `key.path = value` pairs recovered exactly, **with** list position |
| Field precision / recall / F1 | Micro-averaged over `key.path = value` pairs **without** list position, so a correct line item still counts when an earlier one was missed |
| CER / WER | Edit distance over the canonical serialized JSON, pooled across documents; falls back to the raw generation when it is not parseable |
| Evaluation loss | Token-weighted cross-entropy over target tokens only (prompt and image tokens masked) |

Strict (position-aware) precision/recall/F1 and macro-averaged variants are recorded alongside.

No classification accuracy and no confusion matrix: the task emits a variable-shaped tree of
open-vocabulary strings, so neither has a meaningful class set here.

### VRAM — `outputs/<method>/resource_metrics.json`

`torch.cuda.reset_peak_memory_stats()` immediately before the measured phase, then
`max_memory_allocated()` / `max_memory_reserved()` after. The same definition for both methods.
Model loading and quantization are profiled as a **separate** phase so they cannot leak into the
training figure. With no CUDA device the metric is recorded as unavailable with the reason — never
estimated.

### Training time

`time.perf_counter()` around `trainer.train()`, with CUDA synchronized on both sides, plus ISO
start/end timestamps. Model download, loading and quantization are excluded and timed separately.
In-training validation time is measured too, and reported both inside and outside the total.
`steps_per_second` / `samples_per_second` are derived extras, not substitutes for the total.

### Parameters

Counted from the instantiated model, never inferred from the config:

```python
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

with one correction that matters here: bitsandbytes packs two NF4 values per byte, so a naive
`numel()` reports **half** the parameters for a 4-bit model. `src/vlm_ft/resources.py` recovers the
logical count from `quant_state.shape`, so QLoRA and LoRA report the same total. Parameter **count**
and parameter **memory footprint** are stored as separate fields — quantization changes the second,
not the first.

BASE reports `trainable = 0` by definition. At evaluation time adapters are loaded frozen, so a
plain `requires_grad` count would read 0 for a fine-tuned model; the adapter tensors on the
instantiated model are counted instead, and the smoke tests assert this equals what training
reported.

---

## Artifacts

```
outputs/
├── download_manifest.json          resolved model SHA, bytes on disk, split sizes
├── lora/
│   ├── adapter/                    adapter_config.json + weights + processor
│   ├── resource_metrics.json       VRAM, time, parameters, adapter config, environment
│   ├── training_log_history.json   per-step train loss and per-eval validation loss
│   └── run_config.json             fully resolved config + CLI overrides
├── qlora/                          same layout
├── eval/{base,lora,qlora}/
│   ├── metrics.json                all quality metrics + parameters + inference settings
│   └── predictions.jsonl           per-document prediction, reference and per-document scores
└── comparison/
    ├── comparison.json             merged table + analysis + controlled-comparison check
    ├── comparison.csv / .md
    └── loss_curves.json
```

A multi-seed study nests that same layout under `<output_root>/seed<N>/` and adds:

```
outputs/multiseed/
├── multiseed_comparison.json   per-metric mean/std/min/max, paired per-seed differences,
│                               and the controlled-comparison check run for every seed
├── multiseed_summary.csv       statistics plus the raw per-seed values
├── multiseed_summary.md
└── multiseed_loss_curves.json
```

`resource_metrics.json` follows this schema (extra keys are additive):

```json
{
  "method": "lora",
  "gpu": {"name": "...", "total_vram_bytes": 0,
          "peak_memory_allocated_bytes": 0, "peak_memory_reserved_bytes": 0},
  "training": {"duration_seconds": 0.0, "steps": 0, "samples": 0,
               "steps_per_second": null, "samples_per_second": null},
  "parameters": {"total": 0, "trainable": 0, "trainable_ratio": 0.0}
}
```

---

## Configuration

Any value can be overridden without editing a file:

```bash
python scripts/train.py --method lora \
  --set training.num_train_epochs=1 \
  --set data.max_train_samples=200 \
  --set data.image.longest_edge=512
```

Defaults worth knowing:

| Key | Default | Note |
|---|---|---|
| `data.image.longest_edge` | 768 | 2×2 tiles + 1 global = 405 image tokens |
| `data.max_seq_length` | 1536 | Measured: prompt is 735 tokens, longest observed target 480 — nothing truncates |
| `training.gradient_accumulation_steps` | 8 | Effective batch size 8 at `per_device_train_batch_size: 1` |
| `training.optim` | `adamw_torch` | Same for both methods: only adapters are trained, so QLoRA needs no paged optimizer |
| `generation.max_new_tokens` | 512 | Covers the longest observed target |

Truncation and unsupervised examples are counted by the collator and reported in the artifacts, so
silent target loss cannot go unnoticed.

### Raising the image resolution

Increasing `longest_edge` adds image tokens (measured, for a 960x1280 receipt):

| `longest_edge` | Tiles | Prompt tokens | With the longest target (480) | Suggested `max_seq_length` |
|---|---:|---:|---:|---:|
| 768 (default) | 5 | 735 | 1,217 | 1536 |
| 1152 | 10 | 1,191 | 1,673 | 2048 |
| 1536 | 13 | 1,465 | 1,947 | 2048 |
| 1920 | 21 | 2,194 | 2,676 | 3072 |

Change the resolution **in `configs/base.yaml`**, not with a per-command `--set`: forgetting it on any
one of the five commands would leave the preprocessing unmatched and break the comparison. Editing
base.yaml propagates to every condition.

---

## Hardware and runtime

Both methods must fit the GPU for the comparison to mean anything. With the defaults the model is
2.2 B parameters, sequences run to roughly 1 200 tokens, and gradient checkpointing is on.

**The defaults target an A100/L4-class GPU**: `per_device_train_batch_size: 4` with
`gradient_accumulation_steps: 2` (effective batch size 8, unchanged from batch 1 × 8) keeps the
accelerator busy, which a 2.2 B model at batch 1 does not.

| GPU | Notes |
|---|---|
| **A100 40 GB** | Recommended. bf16, and 1,555 GB/s of memory bandwidth — this workload is bandwidth-bound rather than compute-bound, so bandwidth matters more than peak TFLOPS |
| **L4 22.5 GB** | Fine. bf16, but roughly 300 GB/s, so noticeably slower |
| **T4 16 GB** | Works, but has no bf16 (falls back to fp16) and is slow. Add `--set training.per_device_train_batch_size=1 --set training.gradient_accumulation_steps=8` to **both** runs |

VRAM is not the binding constraint at these defaults — even L4 has headroom. A100 is worth choosing
for speed, not for capacity.

- Expect roughly 1–2 hours for both training runs combined on an L4/A100, appreciably longer on a T4.
- On CUDA OOM, lower `data.image.longest_edge` and `data.max_seq_length` — and apply the same change
  to **both** runs.
- Keep `training.gradient_checkpointing: true`. Turning it off is faster but lets activation memory
  dominate the peak, which **shrinks the apparent VRAM saving** of QLoRA — the very thing being
  measured. If you do turn it off, turn it off for both methods and say so.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Covers the metric definitions (including a randomized check of the Levenshtein fast path against the
pure-Python reference), config inheritance and the method guards, the report/table assembly including
every N/A path, the notebook's structure, and a CPU end-to-end run of `train.py` and `evaluate.py`
against a tiny stand-in model.

### What has been verified, and what has not

Verified on this machine (macOS, CPU only): the full LoRA training path, both evaluation paths, all
artifact schemas, the metric implementations, and the notebook's analysis and plotting cells — run
end-to-end against real CORD data using `hf-internal-testing/tiny-random-Idefics3ForConditionalGeneration`
as a stand-in, and against the real SmolVLM processor for the sequence-length budget.

**Not executed here:** the QLoRA branch, which needs CUDA and bitsandbytes; and any real VRAM or
training-time measurement, which needs a GPU. The QLoRA configuration path (`BitsAndBytesConfig` with
NF4 + double quant + matched compute dtype) is unit-tested, and the 4-bit parameter-counting
correction is implemented, but the first real QLoRA run should be a short smoke test
(`--set training.max_steps=4`) before a full run.

Verified dependency versions: transformers 5.17.0, peft 0.20.0, datasets 5.0.1, accelerate 1.15.0,
torch 2.14.0. The code also carries compatibility shims for transformers 4.46+ (`dtype` vs
`torch_dtype`, `AutoModelForImageTextToText` vs `AutoModelForVision2Seq`, and an explicit warmup-step
conversion since transformers 5 removed `warmup_ratio`).

---

## Layout

```
configs/       base.yaml + the two method configs that inherit it
src/vlm_ft/    config, data, metrics, modeling, resources, report, seeding
scripts/       download_model.py, train.py, evaluate.py, run_experiment.py, collect_results.py
tests/         metric, config, report, notebook and CPU end-to-end tests
notebooks/     the Colab comparison notebook
```
