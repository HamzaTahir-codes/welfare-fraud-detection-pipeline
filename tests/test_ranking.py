"""Unit tests for the queue ranking (src/dashboard/ranking.py), the offline re-rank core
(src/dashboard/rerank.py) and the precision@k helper (src/dashboard/evaluation.py),
on tiny hand-built frames."""

import ast
import builtins
import io
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.dashboard import insights, publish
from src.dashboard.evaluation import (
    SENSITIVITY_CONFIGS,
    answer_key_coverage,
    legacy_order,
    planted_ids,
    precision_at_k,
    sensitivity_table,
)
from src.dashboard.publish import load_signal_weights
from src.dashboard.ranking import (
    HEURISTIC_PRIOR,
    HEURISTIC_PRIORS,
    PRECISION_COLUMN,
    QUEUE_COLUMNS,
    VALIDATED_SIGNALS,
    build_case_queue,
    evidence_scores,
    queue_kpis,
    signal_weights,
    weight_map,
    weights_records,
)
from src.dashboard.rerank import rerank_frames, rerank_snapshot

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "dashboard" / "20260922052809"

FAMILY = {
    "shared_bank_account": "payment_channel",
    "duplicate_cycle_payment": "payment_channel",
    "fuzzy_duplicate_identity": "identity",
    "cnic_mismatch": "identity",
    "orphaned_disbursement": "existence",
    "payment_to_inactive_beneficiary": "eligibility",
    "multivariate_anomaly": "statistical",
}


def signals(rows) -> pd.DataFrame:
    """rows: (beneficiary_id, signal_type, score) -> a signals_long frame."""
    df = pd.DataFrame(rows, columns=["beneficiary_id", "signal_type", "score"])
    df["family"] = df["signal_type"].map(FAMILY)
    df["method"] = df["signal_type"].map(
        lambda s: (
            "fuzzy_linkage"
            if s == "fuzzy_duplicate_identity"
            else "isolation_forest"
            if s == "multivariate_anomaly"
            else "rule_based"
        )
    )
    df["n_flags"] = 1
    return df


def metrics_row(detector, precision, n_flagged, strict=None, has_gt=True) -> dict:
    return {
        "detector": detector,
        "has_ground_truth": has_gt,
        "n_flagged": n_flagged,
        "precision": precision,
        "precision_strict": strict,
    }


# ── signal_weights ──────────────────────────────────────────────────────


def test_shared_account_uses_strict_precision_and_is_smoothed():
    # The answer key does not name the hub owner, so the owner must not count as a hit.
    m = pd.DataFrame([metrics_row("rule:shared_bank_account", precision=1.0, n_flagged=98, strict=0.5)])
    w = signal_weights(m).set_index("signal_type")
    # strict 0.5 of 98 = 49 hits -> (49 + 1) / (98 + 2)
    assert w.loc["shared_bank_account", "weight"] == pytest.approx(0.5)
    assert w.loc["shared_bank_account", "source"] == "measured"
    assert w.loc["shared_bank_account", "column"] == "precision_strict"
    assert w.loc["shared_bank_account", "precision"] == pytest.approx(0.5)


def test_fuzzy_uses_plain_precision_because_both_pair_members_are_planted():
    m = pd.DataFrame([metrics_row("fuzzy:duplicate_identity", precision=0.95, n_flagged=80, strict=0.475)])
    w = signal_weights(m).set_index("signal_type")
    assert PRECISION_COLUMN["fuzzy_duplicate_identity"] == "precision"
    assert w.loc["fuzzy_duplicate_identity", "column"] == "precision"
    assert w.loc["fuzzy_duplicate_identity", "weight"] == pytest.approx((76 + 1) / 82)


def test_precision_column_and_priors_can_be_overridden_for_sensitivity_checks():
    m = pd.DataFrame(
        [
            metrics_row("fuzzy:duplicate_identity", precision=0.95, n_flagged=80, strict=0.475),
            metrics_row("rule:shared_bank_account", precision=1.0, n_flagged=98, strict=0.5),
        ]
    )
    w = signal_weights(m, precision_column="precision_strict", priors={}).set_index("signal_type")
    assert w.loc["fuzzy_duplicate_identity", "weight"] == pytest.approx((38 + 1) / 82)
    assert w.loc["cnic_mismatch", "weight"] == HEURISTIC_PRIOR  # flat prior, no override
    w = signal_weights(m, precision_column="precision").set_index("signal_type")
    assert w.loc["shared_bank_account", "weight"] == pytest.approx(99 / 100)
    with pytest.raises(ValueError):
        signal_weights(m, precision_column="recall")


def test_shipped_weights_on_the_tracked_snapshot_metrics():
    m = pd.read_csv(SNAPSHOT / "results" / "comparison_detector_metrics.csv")
    w = signal_weights(m).set_index("signal_type")
    assert w.loc["fuzzy_duplicate_identity", "precision"] == pytest.approx(0.9512, abs=1e-4)
    assert w.loc["shared_bank_account", "precision"] == pytest.approx(0.8489, abs=1e-4)
    assert w.loc["multivariate_anomaly", "weight"] < 0.06
    assert w.loc["cnic_mismatch", "source"] == "prior"


def test_missing_column_falls_back_to_the_other_and_never_reaches_one():
    m = pd.DataFrame([metrics_row("rule:orphaned_disbursement", precision=1.0, n_flagged=25, strict=None)])
    w = signal_weights(m).set_index("signal_type")
    assert w.loc["orphaned_disbursement", "weight"] == pytest.approx(26 / 27, abs=1e-6)
    assert w.loc["orphaned_disbursement", "weight"] < 1.0
    assert w.loc["orphaned_disbursement", "column"] == "precision"


def test_heuristics_get_their_prior_even_when_the_file_lists_a_number():
    m = pd.DataFrame(
        [
            metrics_row("rule:cnic_mismatch", precision=0.9, n_flagged=100, strict=0.9, has_gt=False),
            metrics_row("rule:duplicate_cycle_payment", precision=1.0, n_flagged=100, strict=1.0, has_gt=True),
        ]
    )
    w = signal_weights(m).set_index("signal_type")
    assert w.loc["cnic_mismatch", "weight"] == HEURISTIC_PRIORS["cnic_mismatch"]
    assert w.loc["duplicate_cycle_payment", "weight"] == HEURISTIC_PRIOR
    assert set(w.loc[["cnic_mismatch", "duplicate_cycle_payment"], "source"]) == {"prior"}


def test_missing_or_ungraded_metrics_fall_back_to_the_prior():
    assert set(signal_weights(None)["source"]) == {"prior"}
    m = pd.DataFrame([metrics_row("iforest:layered", precision=0.9, n_flagged=10, has_gt=False)])
    w = signal_weights(m).set_index("signal_type")
    assert w.loc["multivariate_anomaly", "weight"] == HEURISTIC_PRIOR


def test_weights_records_are_json_safe():
    recs = weights_records(signal_weights(None))
    assert all(r["precision"] is None for r in recs)
    assert all(isinstance(r["weight"], float) for r in recs)


# ── evidence_scores ─────────────────────────────────────────────────────


def test_same_family_signals_do_not_double_count():
    s = signals([("A", "shared_bank_account", 0.7), ("A", "duplicate_cycle_payment", 0.75)])
    score = evidence_scores(s, {"shared_bank_account": 0.8, "duplicate_cycle_payment": 0.5})
    assert score["A"] == pytest.approx(0.8)  # best of the family, not 1 - 0.2 * 0.5


def test_independent_families_combine_with_noisy_or():
    s = signals([("A", "shared_bank_account", 0.7), ("A", "fuzzy_duplicate_identity", 0.9)])
    score = evidence_scores(s, {"shared_bank_account": 0.5, "fuzzy_duplicate_identity": 0.5})
    assert score["A"] == pytest.approx(0.75)


def test_unknown_signal_gets_the_prior():
    s = signals([("A", "shared_bank_account", 0.7)]).assign(signal_type="brand_new_rule", family="new")
    assert evidence_scores(s, {})["A"] == pytest.approx(HEURISTIC_PRIOR)


# ── build_case_queue ────────────────────────────────────────────────────

WEIGHTS = {
    "shared_bank_account": 0.85,
    "orphaned_disbursement": 0.96,
    "fuzzy_duplicate_identity": 0.48,
    "multivariate_anomaly": 0.05,
    "cnic_mismatch": 0.01,
    "duplicate_cycle_payment": 0.02,
    "payment_to_inactive_beneficiary": 0.02,
}


def test_noisy_identity_plus_statistical_ranks_below_one_strong_validated_signal():
    # The old order put NOISY first: two families, one of them "validated" (Isolation Forest).
    s = signals(
        [
            ("NOISY", "cnic_mismatch", 0.95),
            ("NOISY", "multivariate_anomaly", 0.70),
            ("STRONG", "shared_bank_account", 0.70),
        ]
    )
    q = build_case_queue(s, {"NOISY", "STRONG"}, WEIGHTS)
    assert q["beneficiary_id"].tolist() == ["STRONG", "NOISY"]
    assert q["rank"].tolist() == [1, 2]


def test_correlated_signals_rank_below_independent_evidence():
    s = signals(
        [
            ("CORR", "shared_bank_account", 0.70),
            ("CORR", "duplicate_cycle_payment", 0.75),
            ("INDEP", "shared_bank_account", 0.70),
            ("INDEP", "orphaned_disbursement", 0.85),
        ]
    )
    q = build_case_queue(s, {"CORR"}, WEIGHTS).set_index("beneficiary_id")
    assert q.loc["INDEP", "rank"] == 1
    assert q.loc["CORR", "evidence_score"] == pytest.approx(0.85)
    assert not q.loc["INDEP", "in_registry"]


def test_ties_keep_the_old_tie_breaks_and_are_deterministic():
    s = signals(
        [
            ("B2", "shared_bank_account", 0.70),
            ("B1", "shared_bank_account", 0.70),
            ("B3", "shared_bank_account", 0.70),
            ("B3", "duplicate_cycle_payment", 0.75),  # same family: same score, higher raw score
        ]
    )
    q = build_case_queue(s, set(), WEIGHTS)
    assert q["evidence_score"].nunique() == 1
    assert q["beneficiary_id"].tolist() == ["B3", "B1", "B2"]


def test_queue_keeps_all_columns_and_one_row_per_beneficiary():
    s = signals(
        [
            ("A", "shared_bank_account", 0.70),
            ("A", "cnic_mismatch", 0.95),
            ("B", "multivariate_anomaly", 0.61),
        ]
    )
    q = build_case_queue(s, {"A", "B"}, WEIGHTS)
    assert list(q.columns) == QUEUE_COLUMNS
    assert len(q) == 2
    a = q.set_index("beneficiary_id").loc["A"]
    assert a["n_families"] == 2 and a["n_validated_families"] == 1
    assert a["signal_types"] == "cnic_mismatch, shared_bank_account"
    assert a["max_score"] == pytest.approx(0.95)


def test_queue_kpis():
    s = signals([("A", "shared_bank_account", 0.7), ("A", "cnic_mismatch", 0.95), ("Z", "orphaned_disbursement", 0.85)])
    k = queue_kpis(build_case_queue(s, {"A"}, WEIGHTS))
    assert k == {"queue_size": 2, "queue_by_families": {1: 1, 2: 1}, "not_in_registry": 1}


# ── rerank core ─────────────────────────────────────────────────────────


def test_rerank_frames_reorders_and_keeps_registry_membership():
    s = signals(
        [
            ("NOISY", "cnic_mismatch", 0.95),
            ("NOISY", "multivariate_anomaly", 0.70),
            ("GHOST", "orphaned_disbursement", 0.85),
        ]
    )
    old = pd.DataFrame({"rank": [1, 2], "beneficiary_id": ["NOISY", "GHOST"], "in_registry": [True, False]})
    m = pd.DataFrame(
        [
            metrics_row("rule:orphaned_disbursement", 1.0, 25, 1.0),
            metrics_row("iforest:layered", 0.05, 400, 0.05),
        ]
    )
    q, w = rerank_frames(s, old, m)
    assert q["beneficiary_id"].tolist() == ["GHOST", "NOISY"]
    assert q.set_index("beneficiary_id")["in_registry"].to_dict() == {"GHOST": False, "NOISY": True}
    assert weight_map(w)["orphaned_disbursement"] == pytest.approx(26 / 27, abs=1e-6)


def test_rerank_frames_refuses_a_different_set_of_beneficiaries():
    s = signals([("A", "shared_bank_account", 0.7)])
    old = pd.DataFrame({"rank": [1], "beneficiary_id": ["B"], "in_registry": [True]})
    with pytest.raises(ValueError):
        rerank_frames(s, old, None)


# ── evaluation (precision@k) ────────────────────────────────────────────


def test_planted_ids_reads_all_three_id_columns():
    key = pd.DataFrame(
        {
            "fraud_type": ["shared_account", "duplicate_identity"],
            "beneficiary_id": ["B1", None],
            "original_beneficiary_id": [None, "B2"],
            "duplicate_beneficiary_id": [None, "B3"],
        }
    )
    assert planted_ids(key) == {"B1", "B2", "B3"}


def test_precision_at_k_follows_rank_and_caps_k():
    q = pd.DataFrame({"rank": [3, 1, 2], "beneficiary_id": ["C", "A", "B"]})
    t = precision_at_k(q, {"A", "C"}, [1, 2, 10]).set_index("k")
    assert t.loc[1, "precision"] == 1.0
    assert t.loc[2, "precision"] == 0.5
    assert t.loc[10, "n_planted"] == 2
    assert t.loc[10, "precision"] == pytest.approx(2 / 3)


def test_answer_key_coverage_flags_a_key_from_another_run():
    q = pd.DataFrame({"rank": [1, 2], "beneficiary_id": ["A", "B"]})
    assert answer_key_coverage(q, {"A", "B"}) == 1.0
    assert answer_key_coverage(q, {"A", "X", "Y", "Z"}) == 0.25
    assert pd.isna(answer_key_coverage(q, set()))


def test_legacy_order_reproduces_the_old_ranking_rule():
    q = pd.DataFrame(
        {
            "rank": [1, 2, 3],
            "beneficiary_id": ["X", "Y", "Z"],
            "n_validated_families": [1, 2, 1],
            "n_families": [1, 2, 2],
            "max_score": [0.9, 0.1, 0.5],
        }
    )
    out = legacy_order(q)
    assert out["beneficiary_id"].tolist() == ["Y", "Z", "X"]
    assert out["rank"].tolist() == [1, 2, 3]


# ── rerank_snapshot file I/O (tiny snapshot in pytest's tmp_path) ────────


@pytest.fixture
def tiny_snapshot(tmp_path):
    """A minimal snapshot in pytest's per-test temp folder (never inside the repo)."""
    d = tmp_path / "snapshot"
    (d / "results").mkdir(parents=True)
    signals(
        [
            ("NOISY", "cnic_mismatch", 0.95),
            ("NOISY", "multivariate_anomaly", 0.70),
            ("GHOST", "orphaned_disbursement", 0.85),
            ("HUB", "shared_bank_account", 0.70),
        ]
    ).to_csv(d / "signals_long.csv", index=False)
    old = pd.DataFrame(
        {
            "rank": [1, 2, 3],
            "beneficiary_id": ["NOISY", "HUB", "GHOST"],
            "n_validated_families": [1, 1, 1],
            "n_families": [2, 1, 1],
            "max_score": [0.95, 0.70, 0.85],
            "in_registry": [True, True, False],
        }
    )
    old.to_csv(d / "case_queue.csv", index=False)
    pd.DataFrame(
        [
            metrics_row("rule:orphaned_disbursement", 1.0, 25, 1.0),
            metrics_row("rule:shared_bank_account", 1.0, 100, 0.85),
            metrics_row("iforest:layered", 0.05, 400, 0.05),
        ]
    ).to_csv(d / "results" / "comparison_detector_metrics.csv", index=False)
    (d / "kpis.json").write_text(json.dumps({"queue_size": 99, "unrelated": "kept"}))
    (d / "manifest.json").write_text(json.dumps({"snapshot_id": "_pytest", "queue_size": 99}))
    return d


def _contents(d: Path) -> dict:
    return {p.relative_to(d): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def test_rerank_snapshot_dry_run_writes_nothing(tiny_snapshot):
    before = _contents(tiny_snapshot)
    q = rerank_snapshot(tiny_snapshot, dry_run=True)
    assert q["beneficiary_id"].tolist() == ["GHOST", "HUB", "NOISY"]
    assert _contents(tiny_snapshot) == before


def test_rerank_snapshot_writes_queue_kpis_and_manifest(tiny_snapshot):
    rerank_snapshot(tiny_snapshot)
    q = pd.read_csv(tiny_snapshot / "case_queue.csv")
    assert list(q.columns) == QUEUE_COLUMNS
    assert q["beneficiary_id"].tolist() == ["GHOST", "HUB", "NOISY"]
    assert q.set_index("beneficiary_id")["in_registry"].to_dict() == {"GHOST": False, "HUB": True, "NOISY": True}

    kpis = json.loads((tiny_snapshot / "kpis.json").read_text())
    assert kpis["unrelated"] == "kept"
    assert kpis["queue_size"] == 3 and kpis["not_in_registry"] == 1
    assert kpis["queue_by_families"] == {"1": 2, "2": 1}
    by_signal = {r["signal_type"]: r for r in kpis["ranking_weights"]}
    assert by_signal["shared_bank_account"]["column"] == "precision_strict"
    assert by_signal["cnic_mismatch"]["source"] == "prior"

    manifest = json.loads((tiny_snapshot / "manifest.json").read_text())
    assert manifest["snapshot_id"] == "_pytest"
    assert manifest["queue_size"] == 3
    assert datetime.fromisoformat(manifest["reranked_at"]).tzinfo is not None
    assert not list(tiny_snapshot.rglob("*.tmp"))

    # Re-running gives a byte-identical case_queue.csv (only manifest's timestamp changes).
    first = (tiny_snapshot / "case_queue.csv").read_bytes()
    rerank_snapshot(tiny_snapshot)
    assert (tiny_snapshot / "case_queue.csv").read_bytes() == first


def test_rerank_snapshot_without_metrics_falls_back_to_the_prior(tiny_snapshot):
    (tiny_snapshot / "results" / "comparison_detector_metrics.csv").unlink()
    rerank_snapshot(tiny_snapshot)
    kpis = json.loads((tiny_snapshot / "kpis.json").read_text())
    assert {r["source"] for r in kpis["ranking_weights"]} == {"prior"}


def test_sensitivity_table_has_the_old_order_and_every_config(tiny_snapshot):
    t = sensitivity_table(tiny_snapshot, {"GHOST", "HUB"}, [1, 2])
    assert len(t) == 1 + len(SENSITIVITY_CONFIGS)
    assert t.loc[0, "ranking"].startswith("before")
    assert t.loc[0, "@1"] == 0.0  # the old order put NOISY first
    assert (t.loc[1:, "@2"] == 1.0).all()


def test_publish_load_signal_weights(tiny_snapshot):
    w = load_signal_weights(tiny_snapshot / "results").set_index("signal_type")
    assert w.loc["shared_bank_account", "source"] == "measured"
    missing = load_signal_weights(tiny_snapshot / "no_such_folder")
    assert set(missing["source"]) == {"prior"}
    assert set(missing["weight"]) <= {HEURISTIC_PRIOR, *HEURISTIC_PRIORS.values()}


def _takeaway(rows, with_score=True):
    q = pd.DataFrame(rows, columns=["rank", "beneficiary_id", "n_validated_families", "evidence_score"])
    if not with_score:
        q = q.drop(columns="evidence_score")
    return insights.queue_takeaway({"queue": q})


def test_queue_takeaway_reports_actual_ranks_not_the_old_rule():
    # Two people with 2 validated families, ranked BELOW a single strong signal.
    text = _takeaway([(1, "GHOST", 1, 0.96), (2, "A", 2, 0.86), (5, "B", 2, 0.80), (3, "C", 1, 0.5)])
    assert "2 beneficiaries" in text
    assert "highest of them is at rank 2 and the lowest at rank 5 (of 4)" in text
    assert "evidence score" in text
    assert "top of the queue" not in text
    assert "1 beneficiary is" in _takeaway([(1, "GHOST", 1, 0.96), (2, "A", 2, 0.86)])


def test_queue_takeaway_without_multi_validated_or_score_column():
    text = _takeaway([(1, "A", 1, 0.9), (2, "B", 0, 0.1)])
    assert text.startswith("No beneficiary")
    assert "evidence score" in text
    old = _takeaway([(1, "A", 1, 0.9)], with_score=False)
    assert "evidence score" not in old and "older snapshot" in old


def test_validated_signal_lists_stay_in_sync():
    assert insights.VALIDATED_SIGNALS == VALIDATED_SIGNALS == publish.VALIDATED_SIGNALS


# ── the answer key never feeds the ranking ──────────────────────────────


def test_ranking_never_opens_the_answer_key(tiny_snapshot, monkeypatch):
    """Behavioural guard: any attempt to open a ground_truth file while ranking fails the test."""
    opened = []
    real_open = builtins.open

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        if "ground_truth" in str(file):
            raise AssertionError(f"ranking code opened the answer key: {file}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    rerank_snapshot(tiny_snapshot)
    load_signal_weights(tiny_snapshot / "results")
    assert any("signals_long.csv" in f for f in opened)  # the spy really saw the reads
    assert not any("ground_truth" in f for f in opened)


def test_ranking_modules_do_not_reference_the_answer_key():
    """Static guard on the parsed code (not only import lines): no import of the evaluation
    helper, no dynamic imports, and no string or name that points at the answer key."""
    root = ROOT / "src" / "dashboard"
    for name in ("ranking.py", "publish.py", "rerank.py"):
        tree = ast.parse((root / name).read_text())
        docstrings = {
            id(n.body[0].value)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
            and n.body
            and isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "evaluation" not in (node.module or ""), name
                assert not any("evaluation" in a.name for a in node.names), name
            elif isinstance(node, ast.Import):
                assert not any("evaluation" in a.name or a.name == "importlib" for a in node.names), name
            elif isinstance(node, ast.Name):
                assert node.id not in ("__import__", "ANSWER_KEY", "planted_ids", "load_planted_ids"), name
            elif isinstance(node, ast.Attribute):
                assert node.attr not in ("import_module", "ANSWER_KEY", "load_planted_ids"), name
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                assert "ground_truth_fraud" not in node.value and "evaluation" not in node.value, name
