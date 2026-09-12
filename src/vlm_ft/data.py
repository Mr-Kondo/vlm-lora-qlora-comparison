"""Dataset loading, prompt/target formatting and batching.

This module is shared verbatim by ``--method lora`` and ``--method qlora`` and
by all three evaluation variants. Nothing here branches on the training method,
which is what makes the LoRA/QLoRA comparison a comparison of the *methods*
rather than of two independently written pipelines.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import metrics

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def load_splits(cfg, splits: Sequence[str] = ("train", "validation", "test")) -> dict:
    """Load the requested CORD splits, applying deterministic subsampling.

    Subsampling uses ``experiment.seed``, so a capped LoRA run and a capped
    QLoRA run see exactly the same examples in the same order.
    """
    from datasets import load_dataset

    seed = int(cfg.experiment.seed)
    data_cfg = cfg.data
    split_names = {
        "train": data_cfg.train_split,
        "validation": data_cfg.validation_split,
        "test": data_cfg.test_split,
    }
    caps = {
        "train": data_cfg.get("max_train_samples"),
        "validation": data_cfg.get("max_eval_samples"),
        "test": data_cfg.get("max_test_samples"),
    }

    out = {}
    for key in splits:
        dataset = load_dataset(
            data_cfg.dataset_id,
            split=split_names[key],
            revision=data_cfg.get("dataset_revision"),
        )
        cap = caps[key]
        if cap is not None and cap < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(int(cap)))
        LOGGER.info("loaded split %s (%s): %d examples", key, split_names[key], len(dataset))
        out[key] = dataset
    return out


def extract_reference(ground_truth: str | dict) -> dict:
    """Turn a CORD ``ground_truth`` record into the canonical target document."""
    raw = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    parse = raw.get("gt_parse", raw) if isinstance(raw, dict) else raw
    document = metrics.canonicalize_document(parse)
    if document is None:
        raise ValueError(f"ground truth did not canonicalize to an object: {str(raw)[:120]}")
    return document


def target_text(ground_truth: str | dict) -> str:
    """The exact string the model is trained to emit."""
    return metrics.serialize(extract_reference(ground_truth))


# --------------------------------------------------------------------------
# Chat formatting
# --------------------------------------------------------------------------
def build_messages(instruction: str, answer: str | None = None) -> list[dict]:
    """One image + instruction turn, optionally followed by the target answer."""
    messages: list[dict] = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def resolve_image_token_id(processor, model=None) -> int | None:
    """Locate the image placeholder token id across processor/model layouts."""
    for candidate in (
        getattr(processor, "image_token_id", None),
        getattr(getattr(model, "config", None), "image_token_id", None),
    ):
        if isinstance(candidate, int):
            return candidate
    token = getattr(processor, "image_token", None)
    if token is not None:
        token_id = processor.tokenizer.convert_tokens_to_ids(str(token))
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
    return None


def configure_processor(processor, cfg):
    """Apply the shared image-resolution settings to a freshly loaded processor."""
    image_cfg = cfg.data.image
    longest_edge = image_cfg.get("longest_edge")
    if longest_edge:
        processor.image_processor.size = {"longest_edge": int(longest_edge)}
    splitting = image_cfg.get("do_image_splitting")
    if splitting is not None:
        processor.image_processor.do_image_splitting = bool(splitting)
    return processor


# --------------------------------------------------------------------------
# Collation
# --------------------------------------------------------------------------
@dataclass
class CollatorStats:
    """Counters surfaced in the run artifacts so silent data loss is visible."""

    examples: int = 0
    truncated: int = 0
    unsupervised: int = 0
    max_length_seen: int = 0

    def as_dict(self) -> dict:
        return {
            "examples_collated": self.examples,
            "truncated_examples": self.truncated,
            "examples_without_supervision": self.unsupervised,
            "max_sequence_length_seen": self.max_length_seen,
        }


@dataclass
class JsonExtractionCollator:
    """Build training batches of (image, instruction) -> JSON target.

    Prompt tokens are masked out of the loss by comparing the tokenization of
    the prompt-only turn with the tokenization of the full conversation and
    masking their longest common prefix. That is model-agnostic and, unlike
    slicing at a hard-coded template offset, stays correct when the tokenizer
    merges a token across the prompt/answer boundary.
    """

    processor: Any
    instruction: str
    max_seq_length: int = 1536
    image_token_id: int | None = None
    label_pad_token_id: int = -100
    stats: CollatorStats = field(default_factory=CollatorStats)

    def _render(self, answer: str | None, add_generation_prompt: bool) -> str:
        return self.processor.apply_chat_template(
            build_messages(self.instruction, answer),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def __call__(self, features: Sequence[dict]) -> dict:
        import torch

        # Right padding keeps prompt tokens at the start of every row, which the
        # prefix masking below relies on. Set per call because evaluate.py shares
        # one processor between this collator and GenerationCollator.
        self.processor.tokenizer.padding_side = "right"

        images, full_texts, prompt_texts = [], [], []
        for feature in features:
            image = feature["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append([image])
            answer = feature.get("target_text") or target_text(feature["ground_truth"])
            full_texts.append(self._render(answer, add_generation_prompt=False))
            prompt_texts.append(self._render(None, add_generation_prompt=True))

        batch = self.processor(
            text=full_texts, images=images, return_tensors="pt", padding=True
        )
        prompt_batch = self.processor(
            text=prompt_texts, images=images, return_tensors="pt", padding=True
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        prompt_lengths = []
        for row in range(input_ids.size(0)):
            full_row = input_ids[row]
            prompt_row = prompt_batch["input_ids"][row]
            prompt_len = int(prompt_batch["attention_mask"][row].sum())
            shared = 0
            limit = min(int(attention_mask[row].sum()), prompt_len)
            while shared < limit and int(full_row[shared]) == int(prompt_row[shared]):
                shared += 1
            prompt_lengths.append(shared)

        batch, attention_mask, prompt_lengths = self._truncate(
            batch, attention_mask, prompt_lengths
        )
        input_ids = batch["input_ids"]

        labels = input_ids.clone()
        labels[attention_mask == 0] = self.label_pad_token_id
        if self.image_token_id is not None:
            labels[input_ids == self.image_token_id] = self.label_pad_token_id
        for row, prompt_len in enumerate(prompt_lengths):
            labels[row, :prompt_len] = self.label_pad_token_id

        self.stats.examples += input_ids.size(0)
        self.stats.max_length_seen = max(self.stats.max_length_seen, int(input_ids.size(1)))
        supervised = (labels != self.label_pad_token_id).sum(dim=1)
        self.stats.unsupervised += int((supervised == 0).sum())

        batch["labels"] = labels
        return batch

    def _truncate(self, batch, attention_mask, prompt_lengths):
        """Right-truncate over-long rows, counting every one that is affected.

        Truncating from the right preserves the image tokens (which sit near the
        start and must stay aligned with ``pixel_values``) and only ever removes
        the tail of the target, which is then reported rather than hidden.
        """
        seq_len = batch["input_ids"].size(1)
        if seq_len <= self.max_seq_length:
            return batch, attention_mask, prompt_lengths

        affected = int((attention_mask.sum(dim=1) > self.max_seq_length).sum())
        self.stats.truncated += affected
        LOGGER.warning(
            "truncating %d/%d example(s) from %d to %d tokens; raise data.max_seq_length "
            "or lower data.image.longest_edge to avoid losing target text",
            affected,
            attention_mask.size(0),
            seq_len,
            self.max_seq_length,
        )
        for key in ("input_ids", "attention_mask"):
            batch[key] = batch[key][:, : self.max_seq_length]
        attention_mask = batch["attention_mask"]
        prompt_lengths = [min(p, self.max_seq_length) for p in prompt_lengths]
        return batch, attention_mask, prompt_lengths


@dataclass
class GenerationCollator:
    """Prompt-only batches for inference. Left padding keeps generations aligned."""

    processor: Any
    instruction: str

    def __call__(self, features: Sequence[dict]) -> dict:
        # Left padding so every row's generation starts at the same offset.
        self.processor.tokenizer.padding_side = "left"
        images, texts = [], []
        for feature in features:
            image = feature["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append([image])
            texts.append(
                self.processor.apply_chat_template(
                    build_messages(self.instruction, None),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return self.processor(text=texts, images=images, return_tensors="pt", padding=True)
