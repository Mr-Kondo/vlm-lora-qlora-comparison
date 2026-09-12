"""Unit tests for the structured-generation metrics."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlm_ft import metrics as M

GT = {
    "menu": [
        {"nm": "J.STB PROMO", "price": "17500"},
        {"nm": "Y.B.BAT", "price": "46000"},
    ],
    "total": {"total_price": "91000", "cashprice": "91000"},
}


# ---------------------------------------------------------------- parsing
def test_extract_plain_json():
    assert M.extract_json_object('{"a": "1"}') == {"a": "1"}


def test_extract_strips_markdown_fence():
    assert M.extract_json_object('```json\n{"a": "1"}\n```') == {"a": "1"}


def test_extract_recovers_from_surrounding_prose():
    text = 'Sure! Here is the receipt:\n{"a": "1"}\nHope that helps.'
    assert M.extract_json_object(text) == {"a": "1"}


def test_extract_rejects_non_object_and_garbage():
    assert M.extract_json_object("[1, 2, 3]") is None
    assert M.extract_json_object("not json at all") is None
    assert M.extract_json_object("") is None
    assert M.extract_json_object(None) is None


# --------------------------------------------------------- canonicalization
def test_single_item_group_becomes_list():
    doc = M.canonicalize_document({"menu": {"nm": "A", "price": "1"}})
    assert doc == {"menu": [{"nm": "A", "price": "1"}]}


def test_keys_sorted_and_whitespace_collapsed():
    doc = M.canonicalize_document({"total": {"b": "  x   y ", "a": 3}})
    assert list(doc["total"].keys()) == ["a", "b"]
    assert doc["total"] == {"a": "3", "b": "x y"}


def test_dict_and_list_forms_of_a_group_are_equivalent():
    a = M.canonicalize_document({"menu": {"nm": "A"}})
    b = M.canonicalize_document({"menu": [{"nm": "A"}]})
    assert a == b


# ------------------------------------------------------------- flattening
def test_flatten_indexes_list_entries():
    pairs = dict(M.flatten(M.canonicalize_document(GT)))
    assert pairs["menu.0.nm"] == "J.STB PROMO"
    assert pairs["menu.1.price"] == "46000"
    assert pairs["total.total_price"] == "91000"


def test_drop_indices():
    assert M.drop_indices("menu.0.nm") == "menu.nm"
    assert M.drop_indices("menu.10.sub.2.nm") == "menu.sub.nm"
    assert M.drop_indices("total.total_price") == "total.total_price"


# ------------------------------------------------------- document scoring
def test_perfect_prediction():
    s = M.score_document(M.serialize(M.canonicalize_document(GT)), GT)
    assert s["json_valid"] and s["exact_match"]
    assert s["char_distance"] == 0 and s["word_distance"] == 0
    assert s["matched"] == s["reference_fields"] == s["predicted_fields"] == 6


def test_key_order_does_not_matter():
    shuffled = {"total": {"cashprice": "91000", "total_price": "91000"}, "menu": GT["menu"]}
    s = M.score_document(json.dumps(shuffled), GT)
    assert s["exact_match"]


def test_reordered_line_items_hurt_strict_but_not_unordered():
    flipped = {"menu": list(reversed(GT["menu"])), "total": GT["total"]}
    s = M.score_document(json.dumps(flipped), GT)
    assert s["matched"] == 6, "unordered field matching should be position-insensitive"
    assert s["matched_strict"] == 2, "strict matching should penalise the shift"
    assert not s["exact_match"]


def test_one_wrong_value():
    wrong = json.loads(json.dumps(GT))
    wrong["total"]["total_price"] = "99000"
    s = M.score_document(json.dumps(wrong), GT)
    assert not s["exact_match"]
    assert s["matched"] == 5 and s["reference_fields"] == 6 and s["predicted_fields"] == 6


def test_invalid_json_scores_zero_fields_but_keeps_cer():
    s = M.score_document("I cannot read this receipt.", GT)
    assert not s["json_valid"] and not s["exact_match"]
    assert s["matched"] == 0 and s["predicted_fields"] == 0
    assert 0 < s["char_distance"] <= max(len(s["reference_text"]), 26) + 26


def test_empty_prediction():
    s = M.score_document("", GT)
    assert not s["json_valid"]
    assert s["char_distance"] == s["char_reference_length"]


# --------------------------------------------------------------- distance
def test_levenshtein_known_values():
    assert M.levenshtein("kitten", "sitting") == 3
    assert M.levenshtein("", "abc") == 3
    assert M.levenshtein("abc", "abc") == 0
    assert M.levenshtein(["a", "b"], ["a", "c"]) == 1


def test_fast_and_reference_levenshtein_agree():
    rng = random.Random(0)
    alphabet = "ab{}\": ,"
    for _ in range(200):
        a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 25)))
        b = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 25)))
        assert M._char_distance(a, b) == M.levenshtein(a, b)


# -------------------------------------------------------------- aggregate
def test_aggregate_micro_math():
    perfect = M.score_document(M.serialize(M.canonicalize_document(GT)), GT)
    empty = M.score_document("nope", GT)
    agg = M.aggregate([perfect, empty])
    assert agg["num_documents"] == 2
    assert agg["json_validity_rate"] == 0.5
    assert agg["document_exact_match"] == 0.5
    # pooled: 6 matched / 6 predicted, 6 matched / 12 reference
    assert agg["field_precision"] == 1.0
    assert agg["field_recall"] == 0.5
    assert abs(agg["field_f1"] - 2 / 3) < 1e-12
    assert agg["field_accuracy"] == agg["field_recall_strict"]


def test_aggregate_all_perfect():
    scores = [M.score_document(M.serialize(M.canonicalize_document(GT)), GT) for _ in range(3)]
    agg = M.aggregate(scores)
    for key in ("json_validity_rate", "document_exact_match", "field_f1", "field_accuracy"):
        assert agg[key] == 1.0
    assert agg["cer"] == 0.0 and agg["wer"] == 0.0


def test_aggregate_requires_documents():
    try:
        M.aggregate([])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ------------------------------------------- repeatable-group normalization
def test_list_valued_group_is_not_double_wrapped():
    doc = M.canonicalize_document({"menu": ["Coke", "Fries"], "total": {"total_price": "5"}})
    assert doc["menu"] == ["Coke", "Fries"]
    paths = [p for p, _ in M.flatten(doc)]
    assert paths == ["menu.0", "menu.1", "total.total_price"]


def test_scalar_group_is_promoted_to_a_list():
    assert M.canonicalize_document({"menu": "Coke"})["menu"] == ["Coke"]


def test_scalar_and_single_element_list_groups_agree():
    assert M.canonicalize_document({"menu": "Coke"}) == M.canonicalize_document({"menu": ["Coke"]})


def test_nested_objects_inside_a_group_survive():
    doc = M.canonicalize_document({"menu": [{"sub": {"b": " x ", "a": "1"}}]})
    assert doc["menu"] == [{"sub": {"a": "1", "b": "x"}}]
    assert dict(M.flatten(doc))["menu.0.sub.b"] == "x"


def test_non_repeatable_group_keeps_its_shape():
    doc = M.canonicalize_document({"total": {"total_price": "5"}})
    assert doc["total"] == {"total_price": "5"}
