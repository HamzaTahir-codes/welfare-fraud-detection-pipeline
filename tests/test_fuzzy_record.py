"""Unit tests for the pure helpers in src/fraud_detection/fuzzy_record.py."""

import pandas as pd
import pytest

from src.fraud_detection.fuzzy_record import dedupe_exact_rows, evaluate_threshold, strip_common_tokens


def test_strip_common_tokens_removes_stopwords():
    assert strip_common_tokens("House No. 12, Street 4", {"house", "no", "street"}) == "12, 4"


def test_strip_common_tokens_falls_back_when_everything_is_stripped():
    assert strip_common_tokens("Street", {"street"}) == "Street"


def test_dedupe_exact_rows_keeps_one_row_per_beneficiary():
    df = pd.DataFrame(
        {
            "beneficiary_id": ["B1", "B1", "B2", "B3"],
            "full_name": ["Ali", "Ali", "Sara", "Copy"],
            "is_duplicate_row": [False, False, False, True],
        }
    )
    out = dedupe_exact_rows(df)
    assert sorted(out.index) == ["B1", "B2"]


def test_evaluate_threshold_precision_recall_f1():
    scored = pd.DataFrame(
        {
            "beneficiary_id_a": ["B1", "B3", "B5"],
            "beneficiary_id_b": ["B2", "B4", "B6"],
            "combined_score": [0.95, 0.90, 0.40],
        }
    )
    truth = {frozenset({"B2", "B1"}), frozenset({"B7", "B8"})}  # pair order must not matter
    m = evaluate_threshold(scored, truth, threshold=0.85)
    assert m["n_flagged"] == 2
    assert m["n_true_positives"] == 1
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_evaluate_threshold_with_nothing_flagged():
    scored = pd.DataFrame({"beneficiary_id_a": ["B1"], "beneficiary_id_b": ["B2"], "combined_score": [0.1]})
    m = evaluate_threshold(scored, {frozenset({"B1", "B2"})}, threshold=0.9)
    assert m["precision"] == m["recall"] == m["f1"] == 0.0
