"""
rule_based.py

Stage 5 - Fraud Detection Layer: Rule-Based Baseline
------------------------------------------------------

Implements six rule-based fraud signal detectors against the cleaned,
Postgres-loaded welfare data (Stage 4 output). Each rule reads from
Postgres into pandas, applies vectorized logic, and the results are
bulk-written back into the `fraud_signals` table.

ENTITY_ID SEMANTICS (important):
entity_id always stores the surrogate `row_id` (cast to text), NOT the
business identifier (beneficiary_id / disbursement_id). This matches the
project's own established principle from Stage 4: natural/business IDs
are not reliably unique once inject_messiness.py duplicates full rows
(a duplicated row keeps the SAME disbursement_id / beneficiary_id as its
source). row_id is the only column guaranteed unique per physical row.

GROUND TRUTH NOTE (for your Stage 6 evaluation writeup):
Only two of these six signal types have a corresponding planted
ground-truth label in `data/ground_truth_fraud_ids.csv`:
    - "shared_bank_account"      <-> ground truth fraud_type "shared_account"
    - "orphaned_disbursement"    <-> ground truth fraud_type "orphan_disbursement"

The remaining four rules (duplicate_cnic_multiple_identities, cnic_mismatch,
duplicate_cycle_payment, payment_to_inactive_beneficiary) are heuristic
signals with no planted ground truth to score against.

Usage:
    python -m src.fraud_detection.rule_based
    python -m src.fraud_detection.rule_based --dry-run
"""

import argparse
import uuid
from datetime import datetime

import pandas as pd
from psycopg2.extras import execute_values

# ASSUMPTION: your Stage 4 connection.py exposes a get_connection() function
# that returns a live psycopg2 connection using the .env-based config you
# already built. Adjust this import path to match your actual module location.
from src.database.connection import get_connection


METHOD = "rule_based"

# Confirmed against your actual cleaned data:
INACTIVE_STATUSES = {"Inactive", "Suspended"}

# Fixed severity scores per rule. Heuristic weights, not probabilities --
# tune once you have real signal volumes and, where available, precision/recall.
SCORES = {
    "duplicate_cnic_multiple_identities": 0.90,
    "shared_bank_account": 0.70,
    "orphaned_disbursement": 0.85,
    "cnic_mismatch": 0.95,
    "duplicate_cycle_payment": 0.75,
    "payment_to_inactive_beneficiary": 0.90,
}


# ---------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------

def read_table(conn, table_name: str) -> pd.DataFrame:
    """Reads a full table into a DataFrame. Fine at current row counts
    (10K / 38K); revisit with explicit column selection if tables grow."""
    query = f"SELECT * FROM {table_name};"
    return pd.read_sql(query, conn)


def write_fraud_signals(conn, df: pd.DataFrame) -> int:
    """Bulk-inserts signal rows into fraud_signals using execute_values,
    mirroring the pattern from your Stage 4 loader_stage.py."""
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


# ---------------------------------------------------------------------
# Rule 1: Duplicate identity -- same CNIC claimed by multiple beneficiary_ids
# ---------------------------------------------------------------------

def rule_duplicate_cnic(df_beneficiaries: pd.DataFrame) -> pd.DataFrame:
    active = df_beneficiaries[
        (df_beneficiaries["is_duplicate_row"] == False)
        & (df_beneficiaries["cnic"].notna())
    ]

    cnic_counts = active.groupby("cnic")["beneficiary_id"].nunique()
    suspicious_cnics = cnic_counts[cnic_counts > 1].index

    flagged = active[active["cnic"].isin(suspicious_cnics)]

    result = flagged[["row_id"]].drop_duplicates().rename(columns={"row_id": "entity_id"})
    result["entity_id"] = result["entity_id"].astype(str)
    result["entity_type"] = "beneficiary"
    result["signal_type"] = "duplicate_cnic_multiple_identities"
    result["score"] = SCORES["duplicate_cnic_multiple_identities"]

    return result


# ---------------------------------------------------------------------
# Rule 2: Shared disbursement channel -- multiple DISTINCT beneficiaries
# funneling into the same bank account. Computed here directly rather than
# trusting disbursements.shared_bank_account_flag, which counts total rows
# per account (>3) and fires for nearly every legitimate multi-month
# beneficiary. See PROJECT_LOG.md for the full writeup of this fix.
# ---------------------------------------------------------------------

def rule_shared_bank_account(df_disbursements: pd.DataFrame) -> pd.DataFrame:
    valid = df_disbursements[df_disbursements["bank_account_number"].notna()]

    account_beneficiary_counts = valid.groupby("bank_account_number")["beneficiary_id"].nunique()
    shared_accounts = account_beneficiary_counts[account_beneficiary_counts > 1].index

    flagged = valid[valid["bank_account_number"].isin(shared_accounts)]

    result = flagged[["row_id"]].drop_duplicates().rename(columns={"row_id": "entity_id"})
    result["entity_id"] = result["entity_id"].astype(str)
    result["entity_type"] = "disbursement"
    result["signal_type"] = "shared_bank_account"
    result["score"] = SCORES["shared_bank_account"]

    return result


# ---------------------------------------------------------------------
# Rule 3: Orphaned disbursement (pass-through of Stage 3 flag)
# ---------------------------------------------------------------------

def rule_orphaned_disbursement(df_disbursements: pd.DataFrame) -> pd.DataFrame:
    flagged = df_disbursements[df_disbursements["orphaned_beneficiary_flag"] == True]

    result = flagged[["row_id"]].drop_duplicates().rename(columns={"row_id": "entity_id"})
    result["entity_id"] = result["entity_id"].astype(str)
    result["entity_type"] = "disbursement"
    result["signal_type"] = "orphaned_disbursement"
    result["score"] = SCORES["orphaned_disbursement"]

    return result


# ---------------------------------------------------------------------
# Rule 4: CNIC mismatch between beneficiaries.cnic and linked national_id_no
# ---------------------------------------------------------------------

def rule_cnic_mismatch(df_beneficiaries: pd.DataFrame, df_national_id: pd.DataFrame) -> pd.DataFrame:
    merged = df_beneficiaries.merge(
        df_national_id,
        left_on="beneficiary_id",
        right_on="linked_beneficiary_id",
        how="inner",
        suffixes=("_ben", "_nid"),
    )

    mismatches = merged[
        merged["national_id_no"].notna() & (merged["cnic"] != merged["national_id_no"])
    ]

    result = mismatches[["row_id"]].drop_duplicates().rename(columns={"row_id": "entity_id"})
    result["entity_id"] = result["entity_id"].astype(str)
    result["entity_type"] = "beneficiary"
    result["signal_type"] = "cnic_mismatch"
    result["score"] = SCORES["cnic_mismatch"]

    return result


# ---------------------------------------------------------------------
# Rule 5: Duplicate cycle payment -- same beneficiary paid twice (two
# DISTINCT disbursement_id events) in one declared disbursement_cycle.
#
# IMPORTANT: inject_messiness.py duplicates ~1.5% of disbursement rows as
# EXACT full-row copies, which keep the SAME disbursement_id as their
# source row. That is Stage 1 data-quality noise, not a second real
# payment. We collapse those artifact-copies down to one event per
# distinct disbursement_id BEFORE checking for genuine same-cycle
# duplicates, so this rule isn't just re-detecting messiness injection.
# ---------------------------------------------------------------------

def rule_duplicate_cycle_payment(df_disbursements: pd.DataFrame) -> pd.DataFrame:
    valid = df_disbursements[
        df_disbursements["beneficiary_id"].notna()
        & df_disbursements["disbursement_cycle"].notna()
    ]

    # Collapse exact full-row duplicate artifacts: a disbursement_id
    # appearing more than once can only happen via inject_messiness.py's
    # row-duplication step (the generator's disb_counter guarantees
    # uniqueness otherwise). Keep one physical row per distinct event.
    unique_events = valid.drop_duplicates(subset=["disbursement_id"])

    event_counts = unique_events.groupby(["beneficiary_id", "disbursement_cycle"])["disbursement_id"].nunique()
    duplicated_pairs = event_counts[event_counts > 1].reset_index()[["beneficiary_id", "disbursement_cycle"]]

    flagged = unique_events.merge(duplicated_pairs, on=["beneficiary_id", "disbursement_cycle"], how="inner")

    result = flagged[["row_id"]].drop_duplicates().rename(columns={"row_id": "entity_id"})
    result["entity_id"] = result["entity_id"].astype(str)
    result["entity_type"] = "disbursement"
    result["signal_type"] = "duplicate_cycle_payment"
    result["score"] = SCORES["duplicate_cycle_payment"]

    return result


# ---------------------------------------------------------------------
# Rule 6: Payment made to a beneficiary whose status is Inactive/Suspended
# ---------------------------------------------------------------------

def rule_inactive_status_payment(
    df_beneficiaries: pd.DataFrame, df_disbursements: pd.DataFrame
) -> pd.DataFrame:
    inactive_ids = df_beneficiaries.loc[
        df_beneficiaries["status_clean"].isin(INACTIVE_STATUSES), "beneficiary_id"
    ].unique()

    flagged = df_disbursements[df_disbursements["beneficiary_id"].isin(inactive_ids)]

    result = flagged[["row_id"]].drop_duplicates().rename(columns={"row_id": "entity_id"})
    result["entity_id"] = result["entity_id"].astype(str)
    result["entity_type"] = "disbursement"
    result["signal_type"] = "payment_to_inactive_beneficiary"
    result["score"] = SCORES["payment_to_inactive_beneficiary"]

    return result


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------

def run_rule_based_detection(dry_run: bool = False) -> dict:
    run_id = str(uuid.uuid4())
    print(f"[rule_based] Starting run {run_id} at {datetime.now().isoformat()}")

    conn = get_connection()

    df_beneficiaries = read_table(conn, "beneficiaries")
    df_national_id = read_table(conn, "national_id_records")
    df_disbursements = read_table(conn, "disbursements")

    print(f"[rule_based] Loaded {len(df_beneficiaries)} beneficiaries, "
          f"{len(df_national_id)} national_id records, "
          f"{len(df_disbursements)} disbursements.")

    signal_frames = [
        rule_duplicate_cnic(df_beneficiaries),
        rule_shared_bank_account(df_disbursements),
        rule_orphaned_disbursement(df_disbursements),
        rule_cnic_mismatch(df_beneficiaries, df_national_id),
        rule_duplicate_cycle_payment(df_disbursements),
        rule_inactive_status_payment(df_beneficiaries, df_disbursements),
    ]

    all_signals = pd.concat(signal_frames, ignore_index=True)
    all_signals["method"] = METHOD
    all_signals["run_id"] = run_id

    summary = all_signals.groupby("signal_type").size().to_dict()

    print("[rule_based] Signal counts by type:")
    for signal_type, count in summary.items():
        print(f"    {signal_type}: {count}")
    print(f"[rule_based] Total signals: {len(all_signals)}")

    if dry_run:
        print("[rule_based] Dry run -- not writing to fraud_signals.")
    else:
        n_written = write_fraud_signals(conn, all_signals)
        print(f"[rule_based] Wrote {n_written} rows to fraud_signals.")

    conn.close()

    return {
        "run_id": run_id,
        "total_signals": len(all_signals),
        "by_signal_type": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rule-based fraud detection (Stage 5).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Compute signals and print summary without writing to the database.")
    args = parser.parse_args()

    run_rule_based_detection(dry_run=args.dry_run)