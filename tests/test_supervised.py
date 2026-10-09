"""Unit tests for src/fraud_detection/supervised.py on tiny hand-built frames (no DB, no data files)."""

import numpy as np
import pandas as pd
import pytest

from src.fraud_detection.isolation_forest import FEATURE_COLUMNS, build_features
from src.fraud_detection.supervised import (
    FEATURE_SETS,
    GROUND_TRUTH_COLUMNS,
    LABEL_PREFIX,
    PRIMARY_TARGET,
    SECONDARY_TARGET,
    add_row_ids,
    assert_no_leakage,
    attach_labels,
    expected_hits_at_k,
    fold_metrics,
    rule_scores,
)


def csv_like_inputs():
    """String-typed frames, as read from the cleaned CSVs (row_id added from file order)."""
    ben = pd.DataFrame(
        {
            "beneficiary_id": ["B1", "B2", "B2", "B3", "B4"],
            "monthly_income_clean": ["5000.0", "0.0", "9999.0", None, "20000.0"],
        }
    )
    disb = pd.DataFrame(
        {
            "disbursement_id": ["D1", "D1", "D2", "D3", "D4", "D5", "D6"],
            "beneficiary_id": ["B1", "B1", "B1", "B2", "B2", "B3", "B9"],
            "amount_pkr_clean": ["6000", "6000", "8000", "7000", "0.1", "5000", "7000"],
        }
    )
    return add_row_ids(ben), disb


def feats_by_id():
    ben, disb = csv_like_inputs()
    return build_features(ben, disb, verbose=False).set_index("beneficiary_id")


def test_add_row_ids_is_one_based_file_order():
    ben, _ = csv_like_inputs()
    assert ben["row_id"].tolist() == [1, 2, 3, 4, 5]


def test_features_collapse_duplicate_disbursement_rows():
    f = feats_by_id()
    # D1 appears twice: one payment event, not two
    assert f.loc["B1", "n_payments"] == 2
    assert f.loc["B1", "mean_amount"] == pytest.approx(7000.0)
    assert f.loc["B1", "total_amount"] == pytest.approx(14000.0)
    assert f.loc["B1", "amount_to_income"] == pytest.approx(7000.0 / 5000.0)


def test_features_first_row_wins_and_income_floor_applies():
    f = feats_by_id()
    # B2 duplicated: lowest row_id (income 0.0) kept; 0.1 amount is a parsing artefact -> ignored
    assert f.loc["B2", "income"] == 0.0
    assert f.loc["B2", "mean_amount"] == pytest.approx(7000.0)
    assert f.loc["B2", "amount_to_income"] == pytest.approx(7000.0 / 1000.0)


def test_features_impute_missing_income_and_skip_unpaid_and_orphans():
    f = feats_by_id()
    assert f.loc["B3", "income"] == pytest.approx(np.median([5000.0, 0.0]))
    assert "B4" not in f.index  # no payments -> not scored
    assert "B9" not in f.index  # orphan disbursement, no beneficiary row
    assert set(FEATURE_COLUMNS) <= set(f.columns)


def test_attach_labels_primary_and_any_planted():
    ben, disb = csv_like_inputs()
    feats = build_features(ben, disb, verbose=False)
    gt = pd.DataFrame(
        {
            "fraud_type": ["multivariate_anomaly", "duplicate_identity", "orphan_disbursement"],
            "bank_account_number": [None, None, "PK1"],
            "beneficiary_id": ["B1", None, "B9"],
            "original_beneficiary_id": [None, "B3", None],
            "duplicate_beneficiary_id": [None, "B77", None],
            "run_id": [None, None, None],
        }
    )
    out = attach_labels(feats, gt).set_index("beneficiary_id")
    assert out[LABEL_PREFIX + PRIMARY_TARGET].to_dict() == {"B1": 1, "B2": 0, "B3": 0}
    assert out[LABEL_PREFIX + SECONDARY_TARGET].to_dict() == {"B1": 1, "B2": 0, "B3": 1}
    # labels are added, never mixed into the feature values
    pd.testing.assert_frame_equal(out[FEATURE_COLUMNS].reset_index(drop=True), feats[FEATURE_COLUMNS])


def test_features_do_not_depend_on_answer_key():
    ben, disb = csv_like_inputs()
    feats = build_features(ben, disb, verbose=False)
    gt_a = pd.DataFrame({"fraud_type": ["multivariate_anomaly"], "beneficiary_id": ["B1"]})
    gt_b = pd.DataFrame({"fraud_type": ["multivariate_anomaly"], "beneficiary_id": ["B2"]})
    xa = attach_labels(feats, gt_a)[FEATURE_COLUMNS]
    xb = attach_labels(feats, gt_b)[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(xa, xb)


def test_leakage_guard_accepts_every_configured_feature_set():
    for cols in FEATURE_SETS.values():
        assert_no_leakage(cols)
        assert not set(cols) & GROUND_TRUTH_COLUMNS
        assert not any(c.startswith(LABEL_PREFIX) for c in cols)


def test_no_count_sets_really_drop_payment_count():
    assert "n_payments" not in FEATURE_SETS["no_count"]
    assert not {"n_payments", "total_amount"} & set(FEATURE_SETS["no_count_no_total"])


@pytest.mark.parametrize(
    "bad",
    [
        "fraud_type",
        "original_beneficiary_id",
        "duplicate_beneficiary_id",
        "run_id",
        "bank_account_number",
        "beneficiary_id",
        "row_id",
        LABEL_PREFIX + PRIMARY_TARGET,
        LABEL_PREFIX + SECONDARY_TARGET,
        "is_planted",  # not a project feature
    ],
)
def test_leakage_guard_rejects_answer_key_labels_ids_and_unknown_columns(bad):
    with pytest.raises(ValueError, match="Leakage guard"):
        assert_no_leakage(["n_payments", bad])


def test_leakage_guard_extra_forbidden():
    with pytest.raises(ValueError, match="Leakage guard"):
        assert_no_leakage(["income"], extra_forbidden={"income"})


def test_expected_hits_splits_ties_at_cutoff():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    # top-2 drawn at random from four tied items holding two positives -> 1 expected hit
    assert expected_hits_at_k(y, s, 2) == pytest.approx(1.0)
    assert expected_hits_at_k(y, s, 4) == pytest.approx(2.0)
    assert expected_hits_at_k(y, np.array([0.9, 0.1, 0.8, 0.2, 0.3]), 2) == pytest.approx(2.0)


def test_fold_metrics_on_perfect_and_rule_scores():
    y = np.array([0] * 98 + [1, 1])
    perfect = np.arange(100, dtype=float)
    m = fold_metrics(y, perfect)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["precision_at_1pct"] == pytest.approx(1.0)  # k=1
    assert m["recall_at_5pct"] == pytest.approx(1.0)  # k=5
    assert m["precision_at_5pct"] == pytest.approx(2 / 5)


def test_rule_scores_threshold():
    assert rule_scores([5, 6, 7, 12]).tolist() == [0.0, 0.0, 1.0, 1.0]
