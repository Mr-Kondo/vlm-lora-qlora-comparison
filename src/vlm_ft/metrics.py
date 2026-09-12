"""Metrics for document-image -> JSON structured generation.

Everything in this module is pure Python (no torch, no transformers) so the
scoring logic can be unit tested independently of any GPU run, and so BASE,
LoRA and QLoRA are scored by byte-identical code.

Deliberately *not* implemented: classification accuracy and confusion matrices.
The task emits a variable-shaped JSON tree with open-vocabulary string values,
so there is no finite class set for which those would be meaningful.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Iterable, Sequence

# CORD groups that hold one *or more* entries. The raw dataset stores a bare
# object when a receipt has a single entry and a list when it has several;
# both sides of every comparison are normalized to a list so that a one-item
# receipt is not scored differently from a two-item one.
REPEATABLE_GROUPS = ("menu", "void_menu")

_WS_RUN = re.compile(r"\s+")
_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)
_INDEX_SEGMENT = re.compile(r"(?:^|\.)\d+(?=\.|$)")

try:  # optional C accelerator; the pure-Python fallback is the reference
    from Levenshtein import distance as _fast_levenshtein
except Exception:  # pragma: no cover - exercised only when the wheel is absent
    _fast_levenshtein = None


# --------------------------------------------------------------------------
# JSON parsing / canonicalization
# --------------------------------------------------------------------------
def extract_json_object(text: str) -> Any | None:
    """Best-effort recovery of a single JSON object from raw model output.

    Applied identically to BASE, LoRA and QLoRA predictions. The leniency
    (stripping markdown fences, taking the outermost balanced ``{...}``) exists
    so that the untuned BASE model is not penalised purely for chattiness; the
    structured content still has to be correct to score.
    """
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    if not candidate:
        return None

    fenced = _FENCE.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    for attempt in (candidate, _outermost_braces(candidate)):
        if attempt is None:
            continue
        try:
            parsed = json.loads(attempt)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _outermost_braces(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def normalize_value(value: Any) -> str:
    """Coerce a leaf to a comparable string: stringify, trim, collapse spaces."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if not isinstance(value, str):
        value = str(value)
    return _WS_RUN.sub(" ", value).strip()


def canonicalize(obj: Any) -> Any:
    """Normalize a parsed JSON subtree into canonical comparison form.

    - object keys are sorted;
    - leaves become whitespace-normalized strings;
    - structure is otherwise preserved.

    Promoting a single-entry repeatable group to a list is done once, at the
    top level, by :func:`canonicalize_document` -- doing it here as well would
    wrap each *element* of an already-list group (``["A", "B"]`` becoming
    ``[["A"], ["B"]]``).

    The same function produces the training target and normalizes predictions,
    so the model is trained on exactly the format it is scored against.
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key in sorted(obj.keys(), key=str):
            out[str(key)] = canonicalize(obj[key])
        return out
    if isinstance(obj, list):
        return [canonicalize(item) for item in obj]
    return normalize_value(obj)


def canonicalize_document(obj: Any) -> dict | None:
    """Canonicalize a whole document, forcing repeatable groups to lists."""
    if not isinstance(obj, dict):
        return None
    out: dict[str, Any] = {}
    for key in sorted(obj.keys(), key=str):
        key = str(key)
        value = canonicalize(obj[key])
        if key in REPEATABLE_GROUPS and not isinstance(value, list):
            value = [value]
        out[key] = value
    return out


def serialize(obj: Any) -> str:
    """Canonical compact JSON text. Used for targets and for CER/WER."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(", ", ": "))


# --------------------------------------------------------------------------
# Field extraction
# --------------------------------------------------------------------------
def flatten(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten a canonical document to ``(dotted.path, value)`` leaf pairs.

    List entries contribute their index, e.g. ``menu.0.nm``.
    """
    pairs: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            pairs.extend(flatten(value, f"{prefix}{key}."))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            pairs.extend(flatten(value, f"{prefix}{idx}."))
    else:
        pairs.append((prefix.rstrip("."), obj if isinstance(obj, str) else normalize_value(obj)))
    return pairs


def drop_indices(path: str) -> str:
    """``menu.0.nm`` -> ``menu.nm`` for order-insensitive field matching."""
    return _INDEX_SEGMENT.sub("", path).strip(".")


def field_pairs(document: Any, ordered: bool) -> Counter:
    """Multiset of field pairs for a document.

    ``ordered=True`` keeps list indices (position-sensitive, strict).
    ``ordered=False`` drops them, which is the conventional key-value
    evaluation for receipt parsing: a correct line item counts as correct even
    if an earlier item was missed and shifted its position.
    """
    if document is None:
        return Counter()
    pairs = flatten(document)
    if ordered:
        return Counter(pairs)
    return Counter((drop_indices(path), value) for path, value in pairs)


# --------------------------------------------------------------------------
# String distance
# --------------------------------------------------------------------------
def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance between two sequences (pure Python reference)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (item_a != item_b),
                )
            )
        previous = current
    return previous[-1]


def _char_distance(pred: str, ref: str) -> int:
    if _fast_levenshtein is not None:
        return _fast_levenshtein(pred, ref)
    return levenshtein(pred, ref)


# --------------------------------------------------------------------------
# Per-document and corpus scoring
# --------------------------------------------------------------------------
def score_document(prediction_text: str, reference: Any) -> dict:
    """Score one generated string against one reference document.

    ``reference`` is the raw (pre-canonicalization) ground-truth object.
    """
    reference_doc = canonicalize_document(reference)
    if reference_doc is None:
        raise ValueError("reference must be a JSON object")
    reference_text = serialize(reference_doc)

    parsed = extract_json_object(prediction_text)
    prediction_doc = canonicalize_document(parsed) if parsed is not None else None
    json_valid = prediction_doc is not None

    # CER/WER fall back to the raw generation when it is not valid JSON, so an
    # unparseable-but-nearly-right output is not scored the same as silence.
    prediction_text_canonical = (
        serialize(prediction_doc) if json_valid else _WS_RUN.sub(" ", (prediction_text or "")).strip()
    )

    result: dict[str, Any] = {
        "json_valid": json_valid,
        "exact_match": json_valid and prediction_doc == reference_doc,
        "reference_text": reference_text,
        "prediction_text_canonical": prediction_text_canonical,
        "char_distance": _char_distance(prediction_text_canonical, reference_text),
        "char_reference_length": len(reference_text),
        "word_distance": levenshtein(prediction_text_canonical.split(), reference_text.split()),
        "word_reference_length": len(reference_text.split()),
    }

    for ordered, suffix in ((False, ""), (True, "_strict")):
        ref_pairs = field_pairs(reference_doc, ordered=ordered)
        pred_pairs = field_pairs(prediction_doc, ordered=ordered)
        matched = sum((ref_pairs & pred_pairs).values())
        result[f"matched{suffix}"] = matched
        result[f"reference_fields{suffix}"] = sum(ref_pairs.values())
        result[f"predicted_fields{suffix}"] = sum(pred_pairs.values())
    return result


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def aggregate(document_scores: Iterable[dict]) -> dict:
    """Pool per-document scores into the corpus-level metric set.

    Precision/recall/F1 are micro-averaged (pooled counts), which is the
    standard for key-value extraction. Macro (per-document mean) variants are
    reported alongside so a few very long receipts cannot dominate the headline.
    """
    scores = list(document_scores)
    n = len(scores)
    if n == 0:
        raise ValueError("no documents to aggregate")

    out: dict[str, Any] = {
        "num_documents": n,
        "json_validity_rate": _safe_div(sum(s["json_valid"] for s in scores), n),
        "document_exact_match": _safe_div(sum(s["exact_match"] for s in scores), n),
    }

    for suffix, label in (("", ""), ("_strict", "_strict")):
        matched = sum(s[f"matched{suffix}"] for s in scores)
        n_ref = sum(s[f"reference_fields{suffix}"] for s in scores)
        n_pred = sum(s[f"predicted_fields{suffix}"] for s in scores)
        precision = _safe_div(matched, n_pred)
        recall = _safe_div(matched, n_ref)
        out[f"field_precision{label}"] = precision
        out[f"field_recall{label}"] = recall
        out[f"field_f1{label}"] = _f1(precision, recall)
        out[f"field_matched{label}"] = matched
        out[f"field_reference_total{label}"] = n_ref
        out[f"field_predicted_total{label}"] = n_pred
        out[f"field_precision{label}_macro"] = _safe_div(
            sum(_safe_div(s[f"matched{suffix}"], s[f"predicted_fields{suffix}"]) for s in scores), n
        )
        out[f"field_recall{label}_macro"] = _safe_div(
            sum(_safe_div(s[f"matched{suffix}"], s[f"reference_fields{suffix}"]) for s in scores), n
        )

    # "Field accuracy" = the fraction of ground-truth fields recovered with the
    # exact value at the exact position. Equal to strict recall by construction;
    # named separately because it is the headline number in the report table.
    out["field_accuracy"] = out["field_recall_strict"]
    out["field_accuracy_macro"] = out["field_recall_strict_macro"]

    out["cer"] = _safe_div(
        sum(s["char_distance"] for s in scores), sum(s["char_reference_length"] for s in scores)
    )
    out["wer"] = _safe_div(
        sum(s["word_distance"] for s in scores), sum(s["word_reference_length"] for s in scores)
    )
    out["cer_macro"] = _safe_div(
        sum(_safe_div(s["char_distance"], s["char_reference_length"]) for s in scores), n
    )
    out["wer_macro"] = _safe_div(
        sum(_safe_div(s["word_distance"], s["word_reference_length"]) for s in scores), n
    )
    return out


#: Metrics shown in the headline comparison table, with display orientation.
#: ``higher_is_better=False`` marks metrics where a lower value is better.
HEADLINE_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("json_validity_rate", "JSON validity rate", True),
    ("document_exact_match", "Document exact match", True),
    ("field_accuracy", "Field accuracy", True),
    ("field_precision", "Field precision", True),
    ("field_recall", "Field recall", True),
    ("field_f1", "Field F1", True),
    ("cer", "CER", False),
    ("wer", "WER", False),
    ("eval_loss", "Evaluation loss", False),
)
