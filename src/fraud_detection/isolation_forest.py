"""
isolation_forest.py

Stage 7 - Fraud Detection Layer: Isolation Forest (multivariate anomaly)
-------------------------------------------------------------------------

Unsupervised detector. It is NEVER shown ground truth: it only sees payment-
behaviour features per beneficiary and scores how easy each beneficiary is to
"isolate" from the crowd (score close to 1 = very unusual, ~0.5 or below =
ordinary). Ground truth is used ONLY in evaluate_isolation_forest.py.

Unit of analysis: one row per beneficiary_id (beneficiary level), matching the
granularity of the planted `multivariate_anomaly` ground truth.

Features (all derived from payment behaviour + stated income):
    n_payments        distinct disbursement_ids paid to the beneficiary
    mean_amount       average payment amount
    total_amount      mean_amount * n_payments (total received)
    income            monthly_income_clean (median-imputed if missing)
    amount_to_income  mean_amount / max(income, INCOME_FLOOR)

Data-quality handling (flag, never drop -- nothing is deleted from the DB):
    * Exact duplicate disbursement rows (same disbursement_id, produced by
      inject_messiness.py) are collapsed BEFORE counting payments. Without this
      a legitimate beneficiary with 6 payments + 1 artifact copy looks like
      "7 payments" and is falsely anomalous.
    * Duplicate beneficiary rows: one row per beneficiary_id is kept (lowest
      row_id). entity_id in fraud_signals = that row_id (as text), same
      convention as Stages 5/6.
    * Beneficiaries with zero payments have no payment behaviour to model, so
      they are not scored (count is logged).
    * Amounts below MIN_PLAUSIBLE_AMOUNT are treated as missing (guard against
      the "Rs. 10,000" -> 0.1 parsing bug in Stage 3's strip_currency_symbols).

Layered mode (--layered):
    Beneficiaries already flagged by the VERIFIED earlier detectors (Stage 5
    shared_bank_account, Stage 6 fuzzy duplicate identity) are left out of the
    model's training AND scoring population for this run (they stay untouched in
    the database -- flag, never drop). Reason: those cases have extreme payment
    counts, which stretch the feature ranges the forest splits over and make
    subtler profiles harder to isolate. Uses earlier detectors' OUTPUT only, never
    ground truth. Requires rule_based.py and fuzzy_record.py to have been run on
    the current data.

Output: one row per FLAGGED beneficiary in fraud_signals with
    method='isolation_forest', signal_type='multivariate_anomaly',
    score = anomaly score in (0,1), plus a run_id.

Usage:
    python -m src.fraud_detection.isolation_forest --dry-run
    python -m src.fraud_detection.isolation_forest --contamination 0.05
    python -m src.fraud_detection.isolation_forest --layered
"""

import argparse
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from sklearn.ensemble import IsolationForest

from src.database.connection import get_connection

METHOD = "isolation_forest"
SIGNAL_TYPE = "multivariate_anomaly"
ENTITY_TYPE = "beneficiary"

FEATURE_COLUMNS = ["n_payments", "mean_amount", "total_amount", "income", "amount_to_income"]

INCOME_FLOOR = 1000          # avoids divide-by-zero for the income=0 tier
MIN_PLAUSIBLE_AMOUNT = 1000  # legit amounts are 5,000-10,000; anything below is a parsing artefact

DEFAULT_CONTAMINATION = 0.05
DEFAULT_N_ESTIMATORS = 300
DEFAULT_SEED = 42


# ---------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------

def load_inputs(conn):
    """Only the columns this stage needs."""
    df_ben = pd.read_sql(
        "SELECT row_id, beneficiary_id, monthly_income_clean FROM beneficiaries;", conn
    )
    df_disb = pd.read_sql(
        "SELECT row_id, disbursement_id, beneficiary_id, amount_pkr_clean FROM disbursements;", conn
    )
    return df_ben, df_disb


def write_fraud_signals(conn, df: pd.DataFrame) -> int:
    """Same 6 columns / execute_values pattern as rule_based.py."""
    if df.empty:
        return 0
    columns = ["entity_id", "entity_type", "method", "signal_type", "score", "run_id"]
    records = list(df[columns].itertuples(index=False, name=None))
    insert_query = """
        INSERT INTO fraud_signals (entity_id, entity_type, method, signal_type, score, run_id)
        VALUES %s;
    """
    with conn.cursor() as cur:
        execute_values(cur, insert_query, records)
    conn.commit()
    return len(records)


def load_covered_ids(conn) -> set:
    """Beneficiaries already flagged by the VERIFIED earlier detectors:
    Stage 5 shared_bank_account (latest rule_based run) and Stage 6 fuzzy duplicate identity."""
    shared = pd.read_sql(
        """
        SELECT DISTINCT d.beneficiary_id
        FROM disbursements d
        JOIN fraud_signals f ON f.entity_id = d.row_id::text
        WHERE f.method = 'rule_based'
          AND f.signal_type = 'shared_bank_account'
          AND f.run_id = (SELECT run_id FROM fraud_signals
                          WHERE method = 'rule_based' ORDER BY created_at DESC LIMIT 1);
        """,
        conn,
    )
    fuzzy = pd.read_sql(
        "SELECT beneficiary_id, matched_record_id FROM record_linkage_matches WHERE is_flagged;", conn
    )
    covered = set(shared["beneficiary_id"].dropna())
    covered |= set(fuzzy["beneficiary_id"].dropna()) | set(fuzzy["matched_record_id"].dropna())
    return covered


def exclude_covered(feats: pd.DataFrame, covered_ids: set) -> pd.DataFrame:
    """Layered mode: drop already-covered beneficiaries from THIS RUN's population only."""
    mask = feats["beneficiary_id"].isin(covered_ids)
    print(f"[isolation_forest] Layered mode: {int(mask.sum())} beneficiaries already flagged by "
          f"Stage 5/6 excluded from training and scoring; {int((~mask).sum())} remain.")
    return feats[~mask].reset_index(drop=True)


# ---------------------------------------------------------------------
# Feature engineering (pure function: DataFrames in, DataFrame out)
# ---------------------------------------------------------------------

def build_features(df_ben: pd.DataFrame, df_disb: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    # One physical row per beneficiary_id (lowest row_id) -- kept, not deleted, in the DB.
    ben = (
        df_ben[df_ben["beneficiary_id"].notna()]
        .sort_values("row_id")
        .drop_duplicates(subset="beneficiary_id", keep="first")
    )

    # Collapse exact duplicate-row artefacts: same disbursement_id == same payment event.
    disb = df_disb[df_disb["beneficiary_id"].notna() & df_disb["disbursement_id"].notna()]
    n_before = len(disb)
    disb = disb.drop_duplicates(subset="disbursement_id", keep="first").copy()

    amount = pd.to_numeric(disb["amount_pkr_clean"], errors="coerce").astype(float)
    n_suspect = int((amount < MIN_PLAUSIBLE_AMOUNT).sum())
    amount = amount.mask(amount < MIN_PLAUSIBLE_AMOUNT)
    disb["amount"] = amount

    agg = disb.groupby("beneficiary_id").agg(
        n_payments=("disbursement_id", "nunique"),
        mean_amount=("amount", "mean"),
    ).reset_index()

    feats = ben[["row_id", "beneficiary_id", "monthly_income_clean"]].merge(
        agg, on="beneficiary_id", how="inner"
    )

    # Imputation (logged): missing/invalid income -> median; no valid amount -> median.
    income = pd.to_numeric(feats["monthly_income_clean"], errors="coerce").astype(float)
    n_income_imputed = int(income.isna().sum())
    feats["income"] = income.fillna(income.median())

    n_amount_imputed = int(feats["mean_amount"].isna().sum())
    feats["mean_amount"] = feats["mean_amount"].fillna(feats["mean_amount"].median())

    feats["total_amount"] = feats["mean_amount"] * feats["n_payments"]
    feats["amount_to_income"] = feats["mean_amount"] / np.maximum(feats["income"], INCOME_FLOOR)

    feats = feats.drop(columns=["monthly_income_clean"]).reset_index(drop=True)

    if verbose:
        print(f"[isolation_forest] Beneficiaries (deduped): {len(ben)}; "
              f"scored (>=1 payment): {len(feats)}; not scored (no payments): {len(ben) - len(feats)}")
        print(f"[isolation_forest] Disbursement rows: {n_before} -> {len(disb)} after collapsing "
              f"exact duplicate disbursement_ids")
        print(f"[isolation_forest] Imputed income for {n_income_imputed} beneficiaries, "
              f"mean_amount for {n_amount_imputed}")
        if n_suspect:
            print(f"[isolation_forest] WARNING: {n_suspect} amounts < {MIN_PLAUSIBLE_AMOUNT} treated as "
                  f"missing. This is the Stage 3 'Rs. 10,000' -> 0.1 parsing bug; fix "
                  f"strip_currency_symbols and re-run ETL for cleaner inputs.")
    return feats


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def fit_and_score(
    feats: pd.DataFrame,
    contamination: float = DEFAULT_CONTAMINATION,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Returns feats plus `anomaly_score` (higher = more anomalous, ~(0,1)) and `is_flagged`.

    NOTE: `contamination` only moves the flagging cut-off; the trees and scores are identical
    for any contamination value. That is what lets the evaluation script sweep it cheaply.
    """
    X = feats[FEATURE_COLUMNS].to_numpy(dtype=float)
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X)

    out = feats.copy()
    out["anomaly_score"] = -model.score_samples(X)   # paper's s(x): 1 = anomalous, <0.5 = normal
    out["is_flagged"] = model.predict(X) == -1
    return out


def to_signal_rows(scored: pd.DataFrame, run_id: str) -> pd.DataFrame:
    flagged = scored[scored["is_flagged"]]
    return pd.DataFrame({
        "entity_id": flagged["row_id"].astype(int).astype(str),
        "entity_type": ENTITY_TYPE,
        "method": METHOD,
        "signal_type": SIGNAL_TYPE,
        "score": flagged["anomaly_score"].round(4),
        "run_id": run_id,
    })


def print_flagged_profile(scored: pd.DataFrame):
    """Plain-language 'why': how flagged beneficiaries differ from everyone else."""
    prof = scored.groupby("is_flagged")[FEATURE_COLUMNS].mean().round(2)
    prof.index = prof.index.map({True: "flagged", False: "not flagged"})
    print("[isolation_forest] Mean feature values, flagged vs not flagged:")
    print(prof.to_string())


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------

def run_isolation_forest(
    contamination: float = DEFAULT_CONTAMINATION,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    seed: int = DEFAULT_SEED,
    dry_run: bool = False,
    export_scores: str = None,
    layered: bool = False,
) -> dict:
    run_id = str(uuid.uuid4())
    print(f"[isolation_forest] Starting run {run_id} at {datetime.now().isoformat()}")
    print(f"[isolation_forest] contamination={contamination}, n_estimators={n_estimators}, "
          f"seed={seed}, layered={layered}")

    conn = get_connection()
    try:
        df_ben, df_disb = load_inputs(conn)
        feats = build_features(df_ben, df_disb)
        if layered:
            feats = exclude_covered(feats, load_covered_ids(conn))
        scored = fit_and_score(feats, contamination, n_estimators, seed)

        signals = to_signal_rows(scored, run_id)
        print(f"[isolation_forest] Flagged {len(signals)} of {len(scored)} scored beneficiaries "
              f"({len(signals) / len(scored):.1%})")
        print_flagged_profile(scored)

        if export_scores:
            scored.to_csv(export_scores, index=False)
            print(f"[isolation_forest] Wrote all scores to {export_scores}")

        if dry_run:
            print("[isolation_forest] Dry run -- not writing to fraud_signals.")
        else:
            n = write_fraud_signals(conn, signals)
            print(f"[isolation_forest] Wrote {n} rows to fraud_signals.")
    finally:
        conn.close()

    return {"run_id": run_id, "n_scored": len(scored), "n_flagged": len(signals)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Isolation Forest anomaly detection (Stage 7).")
    parser.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION,
                        help="Expected share of anomalous beneficiaries (sets the flagging cut-off only).")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="Score and print a summary without writing to fraud_signals.")
    parser.add_argument("--export-scores", type=str, default=None,
                        help="Optional CSV path for all beneficiary scores (handy for dashboard/charts).")
    parser.add_argument("--layered", action="store_true",
                        help="Exclude beneficiaries already flagged by Stage 5/6 before fitting and scoring.")
    args = parser.parse_args()

    run_isolation_forest(args.contamination, args.n_estimators, args.seed, args.dry_run,
                         args.export_scores, args.layered)