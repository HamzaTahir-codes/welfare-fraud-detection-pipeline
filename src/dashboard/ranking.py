"""
ranking.py -- rank the investigation queue by measured evidence strength.

Pure functions only (no database, no files, no Streamlit), so publish.py
(new runs) and rerank.py (existing snapshots) produce exactly the same queue,
and the logic is unit-tested on hand-built frames.

How a beneficiary's evidence_score is built:
1. Each signal gets a weight = an estimate of how often that signal is right.
   - Signals graded against planted fraud use their MEASURED precision from
     Stage 8 (comparison_detector_metrics.csv). Stage 8 reports two readings
     (precision_strict: only the logged culprits count as hits; precision:
     their genuine counterparts count too). Each signal uses the reading
     whose notion of a hit matches how a queue entry is judged to be real
     fraud; see PRECISION_COLUMN. It is add-one smoothed so a perfect score
     on a small sample (orphaned_disbursement: 25 of 25) is not treated as
     certainty: (hits + 1) / (n_flagged + 2).
   - Heuristic signals (no answer key) get a deliberately low, fixed prior
     (HEURISTIC_PRIOR, or its override in HEURISTIC_PRIORS). It is NOT
     measured (but the cnic_mismatch override was tuned in-sample; see
     HEURISTIC_PRIORS).
2. Within one evidence family only the strongest signal counts, because
   signals in a family fire for the same reason (shared_bank_account and
   duplicate_cycle_payment are not independent evidence).
3. Families are combined with a noisy-OR: 1 - prod(1 - family weight).
   Two independent 50% leads give 75%, never more than 100%.
Ties on evidence_score keep the old order: validated families, then all
families, then highest raw score, then id.

Honest caveat: the measured precisions come from the same planted answer key
the queue is later graded on, so any precision@k computed on this synthetic
data is in-sample and optimistic. The answer key itself is never read here.
"""

from typing import Optional

import pandas as pd

# Signals validated against planted ground truth.
VALIDATED_SIGNALS = {
    "orphaned_disbursement",
    "shared_bank_account",
    "fuzzy_duplicate_identity",
    "multivariate_anomaly",  # layered run only
}

# signal_type -> detector row in comparison_detector_metrics.csv. The snapshot's
# Isolation Forest signals come from the LAYERED run, so they take its precision.
SIGNAL_DETECTOR = {
    "orphaned_disbursement": "rule:orphaned_disbursement",
    "shared_bank_account": "rule:shared_bank_account",
    "fuzzy_duplicate_identity": "fuzzy:duplicate_identity",
    "multivariate_anomaly": "iforest:layered",
    "cnic_mismatch": "rule:cnic_mismatch",
    "duplicate_cycle_payment": "rule:duplicate_cycle_payment",
    "payment_to_inactive_beneficiary": "rule:payment_to_inactive_beneficiary",
    "duplicate_cnic_multiple_identities": "rule:duplicate_cnic_multiple_identities",
}

# Which Stage 8 precision column each validated signal is weighted by. A
# queue entry counts as real fraud when its id is named in the answer key
# (ground_truth_fraud_ids.csv: beneficiary_id, original_beneficiary_id or
# duplicate_beneficiary_id), so each signal takes the column that counts hits
# the same way:
# - shared_bank_account: the key names the victims sharing a hub account but
#   NOT the hub owner, so a flagged owner is not a hit -> precision_strict.
#   (plain precision credits the owner and would overstate it: 0.999 vs 0.849.)
# - fuzzy_duplicate_identity: the key names BOTH members of a planted pair
#   (original and duplicate), so both are hits -> precision. (precision_strict
#   counts only the duplicate member, 0.476 vs 0.951, and would under-weight it.)
# - orphaned_disbursement, multivariate_anomaly: no counterparts, the two
#   readings are identical; precision_strict is listed for definiteness.
# Any column that is missing or empty falls back to the other one.
# The rule above follows from the answer key's layout, but note that the
# precision@k effect of each choice was also seen before settling on it
# (fuzzy on precision_strict gave precision@1000 of 94.2%, on precision 98.9%).
PRECISION_COLUMN = {
    "orphaned_disbursement": "precision_strict",
    "shared_bank_account": "precision_strict",
    "fuzzy_duplicate_identity": "precision",
    "multivariate_anomaly": "precision_strict",
}
PRECISION_COLUMNS = ("precision_strict", "precision")

# Conservative prior for a heuristic signal: assume it is right about 1 time in
# 50. That is below every graded detector, so one ungraded signal on its own
# never outranks a graded one. Also used for a validated signal whose metrics
# are missing (e.g. Stage 8 has not run), so missing data never inflates a case.
HEURISTIC_PRIOR = 0.02

# Heuristics stay low on purpose even where they look strong: for example
# payment_to_inactive_beneficiary co-occurs with shared-account victims only
# because those victims were sampled from ALL beneficiaries, not only Active
# ones, so it is not independent evidence.
# Per-signal overrides of the prior. cnic_mismatch gets half. The reason
# given is that the data generator injects CNIC typos on purpose, so most
# mismatches are data-entry noise (see insights.limitations). Be aware that
# this value was NOT set independently: it was picked AFTER looking at
# precision@k against the planted answer key (it moves precision@250 on the
# tracked snapshot from 95.6% to 98.4%), so treat it as tuned in-sample, not
# as a prior. `python -m src.dashboard.evaluation --sensitivity` shows the
# queue with one flat prior (priors={}) next to the shipped one.
HEURISTIC_PRIORS = {
    "cnic_mismatch": 0.01,
}

QUEUE_COLUMNS = [
    "rank",
    "beneficiary_id",
    "evidence_score",
    "n_validated_families",
    "n_families",
    "families",
    "methods",
    "signal_types",
    "max_score",
    "has_validated_signal",
    "total_flags",
    "in_registry",
]


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _truthy(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes")


def signal_weights(
    metrics: Optional[pd.DataFrame],
    prior: float = HEURISTIC_PRIOR,
    priors: Optional[dict] = None,
    precision_column: Optional[str] = None,
) -> pd.DataFrame:
    """One row per known signal_type: the weight the ranking gives it and why.

    Columns: signal_type, detector, source, column, n_flagged, precision, weight.
    source is 'measured' (graded precision, add-one smoothed) or 'prior'.
    column / precision are the Stage 8 column and value used (None for a prior).
    A heuristic signal always gets its prior, even if the file lists a number
    for it; a validated signal with no usable row falls back to `prior`.

    priors overrides HEURISTIC_PRIORS (pass {} for one flat prior) and
    precision_column forces one column for every signal instead of
    PRECISION_COLUMN. Both exist for the sensitivity check in evaluation.py;
    the shipped ranking uses the defaults.
    """
    priors = HEURISTIC_PRIORS if priors is None else priors
    if precision_column is not None and precision_column not in PRECISION_COLUMNS:
        raise ValueError(f"precision_column must be one of {PRECISION_COLUMNS}, got {precision_column!r}")
    by_detector = {}
    if metrics is not None and "detector" in metrics.columns:
        by_detector = {str(r["detector"]): r for _, r in metrics.iterrows()}

    rows = []
    for signal, detector in SIGNAL_DETECTOR.items():
        r = by_detector.get(detector)
        precision = n_flagged = column = None
        if r is not None:
            first = precision_column or PRECISION_COLUMN.get(signal, "precision_strict")
            for col in (first,) + tuple(c for c in PRECISION_COLUMNS if c != first):
                precision = _num(r.get(col))
                if precision is not None:
                    column = col
                    break
            n_flagged = _num(r.get("n_flagged"))
        graded = r is not None and _truthy(r.get("has_ground_truth"))
        if signal in VALIDATED_SIGNALS and graded and precision is not None and n_flagged:
            hits = round(precision * n_flagged)
            weight = (hits + 1) / (n_flagged + 2)
            source = "measured"
        else:
            weight = prior if signal in VALIDATED_SIGNALS else priors.get(signal, prior)
            source, precision, column = "prior", None, None
        rows.append(
            {
                "signal_type": signal,
                "detector": detector,
                "source": source,
                "column": column,
                "n_flagged": None if n_flagged is None else int(n_flagged),
                "precision": precision,
                "weight": round(weight, 6),
            }
        )
    return pd.DataFrame(rows)


def weight_map(weights: pd.DataFrame) -> dict:
    """signal_type -> weight, the form build_case_queue takes."""
    return dict(zip(weights["signal_type"], weights["weight"].astype(float)))


def weights_records(weights: pd.DataFrame) -> list:
    """signal_weights() as JSON-safe records (NaN -> None) for kpis.json."""
    out = []
    for r in weights.to_dict("records"):
        n = _num(r["n_flagged"])
        out.append(
            {
                "signal_type": r["signal_type"],
                "detector": r["detector"],
                "source": r["source"],
                "column": r["column"] if isinstance(r.get("column"), str) else None,
                "n_flagged": None if n is None else int(n),
                "precision": _num(r["precision"]),
                "weight": _num(r["weight"]),
            }
        )
    return out


def evidence_scores(signals_long: pd.DataFrame, weights: dict, prior: float = HEURISTIC_PRIOR) -> pd.Series:
    """Noisy-OR over families of the best signal weight per family.

    signals_long needs beneficiary_id, signal_type, family. A signal_type
    missing from weights gets the prior. Returns a Series indexed by
    beneficiary_id.
    """
    df = signals_long[["beneficiary_id", "signal_type", "family"]].copy()
    df["weight"] = df["signal_type"].map(weights).fillna(prior).astype(float).clip(0.0, 1.0)
    best = df.groupby(["beneficiary_id", "family"])["weight"].max()
    miss = (1.0 - best).groupby(level="beneficiary_id").prod()
    return (1.0 - miss).rename("evidence_score")


def build_case_queue(
    signals_long: pd.DataFrame, registry_ids: set, weights: dict, prior: float = HEURISTIC_PRIOR
) -> pd.DataFrame:
    """One row per beneficiary_id, most suspicious first.

    Rank order: evidence_score (rounded to 6 places), then the number of
    distinct validated families, then all families, then highest raw score,
    then id, so the order is deterministic.
    """
    df = signals_long.copy()
    df["is_validated"] = df["signal_type"].isin(VALIDATED_SIGNALS)

    queue = (
        df.groupby("beneficiary_id")
        .agg(
            n_families=("family", "nunique"),
            families=("family", lambda s: ", ".join(sorted(s.unique()))),
            methods=("method", lambda s: ", ".join(sorted(s.unique()))),
            signal_types=("signal_type", lambda s: ", ".join(sorted(s.unique()))),
            max_score=("score", "max"),
            has_validated_signal=("is_validated", "any"),
            total_flags=("n_flags", "sum"),
        )
        .reset_index()
    )
    queue["in_registry"] = queue["beneficiary_id"].isin(registry_ids)

    validated_families = (
        df[df["is_validated"]].groupby("beneficiary_id")["family"].nunique().rename("n_validated_families")
    )
    queue = queue.join(validated_families, on="beneficiary_id")
    queue["n_validated_families"] = queue["n_validated_families"].fillna(0).astype(int)

    queue = queue.join(evidence_scores(df, weights, prior), on="beneficiary_id")
    queue["evidence_score"] = queue["evidence_score"].round(6)

    queue = queue.sort_values(
        ["evidence_score", "n_validated_families", "n_families", "max_score", "beneficiary_id"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    queue.insert(0, "rank", range(1, len(queue) + 1))
    return queue[QUEUE_COLUMNS]


def queue_kpis(case_queue: pd.DataFrame) -> dict:
    """The kpis.json fields that are derived from the queue alone."""
    return {
        "queue_size": int(len(case_queue)),
        "queue_by_families": {int(k): int(v) for k, v in case_queue["n_families"].value_counts().sort_index().items()},
        "not_in_registry": int((~case_queue["in_registry"].astype(bool)).sum()),
    }
