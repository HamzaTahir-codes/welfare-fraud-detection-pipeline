"""
evaluate_rule_based.py

Stage 5 - Rule-Based Evaluation Against Planted Ground Truth
--------------------------------------------------------------

Scores rule-based fraud signals against data/ground_truth_fraud_ids.csv
for the two signal types that have verifiable ground truth:
    - orphaned_disbursement  <-> ground truth fraud_type "orphan_disbursement"
    - shared_bank_account    <-> ground truth fraud_type "shared_account"

ENTITY_ID SEMANTICS: fraud_signals.entity_id stores the surrogate row_id
(as text), not the business ID. All joins here go through row_id ->
beneficiary_id / disbursement_id via the disbursements table.

GRANULARITY NOTE:
- orphan_disbursement ground truth is 1:1 with a single disbursement row ->
  evaluated at the DISBURSEMENT (row_id) level.
- shared_account ground truth is logged per (hub_account, victim) pair, and
  ONLY victims are logged -- the hub account owner's own beneficiary_id
  never appears in ground truth, even though their disbursements are
  correctly flagged (their account really is shared). Evaluated at the
  BENEFICIARY level, with hub-owner false positives reported separately.

Also reports the overlap between shared_bank_account and
duplicate_cycle_payment signals.

Usage:
    python -m src.fraud_detection.evaluate_rule_based
    python -m src.fraud_detection.evaluate_rule_based --ground-truth data/ground_truth_fraud_ids.csv
"""

import argparse

import pandas as pd

# ASSUMPTION: same connection module used in rule_based.py. Adjust if needed.
from src.database.connection import get_connection


def load_ground_truth(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def load_latest_run_signals(conn) -> pd.DataFrame:
    """Pulls fraud_signals from only the most recent rule_based run, so
    re-running the detector multiple times doesn't double-count signals
    from older runs still sitting in the table."""
    query = """
        SELECT * FROM fraud_signals
        WHERE method = 'rule_based'
          AND run_id = (
              SELECT run_id FROM fraud_signals
              WHERE method = 'rule_based'
              ORDER BY created_at DESC
              LIMIT 1
          );
    """
    df = pd.read_sql(query, conn)
    df["entity_id"] = df["entity_id"].astype(str)
    return df


def load_disbursements(conn) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT row_id, disbursement_id, beneficiary_id, bank_account_number FROM disbursements;",
        conn,
    )
    df["row_id"] = df["row_id"].astype(str)
    return df


def precision_recall_f1(predicted: set, actual: set) -> dict:
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_orphaned_disbursement(
    signals: pd.DataFrame, ground_truth: pd.DataFrame, disbursements: pd.DataFrame
) -> dict:
    gt_orphan = ground_truth[ground_truth["fraud_type"] == "orphan_disbursement"]

    # Map fake beneficiary_id -> row_id via the disbursements table.
    # 1:1 relationship -- plant_orphan_disbursements creates exactly one
    # disbursement row per ground truth entry.
    actual_row_ids = set(
        disbursements.loc[
            disbursements["beneficiary_id"].isin(gt_orphan["beneficiary_id"]), "row_id"
        ]
    )

    predicted_row_ids = set(
        signals.loc[signals["signal_type"] == "orphaned_disbursement", "entity_id"]
    )

    return precision_recall_f1(predicted_row_ids, actual_row_ids)


def evaluate_shared_account(
    signals: pd.DataFrame, ground_truth: pd.DataFrame, disbursements: pd.DataFrame
) -> dict:
    gt_shared = ground_truth[ground_truth["fraud_type"] == "shared_account"]
    actual_beneficiary_ids = set(gt_shared["beneficiary_id"])

    flagged_row_ids = set(
        signals.loc[signals["signal_type"] == "shared_bank_account", "entity_id"]
    )

    predicted_beneficiary_ids = set(
        disbursements.loc[disbursements["row_id"].isin(flagged_row_ids), "beneficiary_id"]
    )

    result = precision_recall_f1(predicted_beneficiary_ids, actual_beneficiary_ids)

    # Diagnostic: how many "false positives" are actually hub account owners?
    # They're conceptually part of the fraud but never logged as victims in
    # ground truth, so they inflate FP count without being a genuine error.
    hub_accounts = set(gt_shared["bank_account_number"])
    false_positive_ids = predicted_beneficiary_ids - actual_beneficiary_ids
    fp_on_hub_account = disbursements.loc[
        disbursements["beneficiary_id"].isin(false_positive_ids)
        & disbursements["bank_account_number"].isin(hub_accounts),
        "beneficiary_id",
    ].unique()

    result["fp_likely_hub_owners"] = len(fp_on_hub_account)
    return result


def analyze_signal_overlap(signals: pd.DataFrame) -> dict:
    """Cross-tab of row_ids flagged by BOTH shared_bank_account and
    duplicate_cycle_payment."""
    shared_ids = set(signals.loc[signals["signal_type"] == "shared_bank_account", "entity_id"])
    duplicate_ids = set(signals.loc[signals["signal_type"] == "duplicate_cycle_payment", "entity_id"])

    overlap = shared_ids & duplicate_ids
    union = shared_ids | duplicate_ids

    return {
        "shared_bank_account_count": len(shared_ids),
        "duplicate_cycle_payment_count": len(duplicate_ids),
        "overlap_count": len(overlap),
        "overlap_pct_of_duplicate_cycle": (len(overlap) / len(duplicate_ids) * 100) if duplicate_ids else 0.0,
        "jaccard_similarity": (len(overlap) / len(union)) if union else 0.0,
    }


def print_report(name: str, metrics: dict):
    print(f"\n[{name}]")
    print(f"    TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")
    if "fp_likely_hub_owners" in metrics:
        print(
            f"    (of {metrics['fp']} false positives, {metrics['fp_likely_hub_owners']} are likely "
            f"hub-account owners not logged as victims in ground truth -- see module docstring)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate rule-based fraud signals against planted ground truth."
    )
    parser.add_argument("--ground-truth", type=str, default="data/ground_truth_fraud_ids.csv")
    args = parser.parse_args()

    conn = get_connection()

    ground_truth = load_ground_truth(args.ground_truth)
    signals = load_latest_run_signals(conn)
    disbursements = load_disbursements(conn)

    if signals.empty:
        print("No rule_based signals found in fraud_signals. Run rule_based.py first (without --dry-run).")
        conn.close()
        return

    print(f"Evaluating run_id: {signals['run_id'].iloc[0]}")
    print(
        f"Ground truth entries: {len(ground_truth)} "
        f"({(ground_truth['fraud_type'] == 'orphan_disbursement').sum()} orphan, "
        f"{(ground_truth['fraud_type'] == 'shared_account').sum()} shared_account)"
    )

    orphan_metrics = evaluate_orphaned_disbursement(signals, ground_truth, disbursements)
    print_report("orphaned_disbursement (row-level)", orphan_metrics)

    shared_metrics = evaluate_shared_account(signals, ground_truth, disbursements)
    print_report("shared_bank_account (beneficiary-level)", shared_metrics)

    overlap = analyze_signal_overlap(signals)
    print("\n[signal overlap: shared_bank_account vs duplicate_cycle_payment]")
    print(f"    shared_bank_account signals:     {overlap['shared_bank_account_count']}")
    print(f"    duplicate_cycle_payment signals: {overlap['duplicate_cycle_payment_count']}")
    print(f"    overlapping row_ids:             {overlap['overlap_count']}")
    print(f"    overlap as % of duplicate_cycle_payment: {overlap['overlap_pct_of_duplicate_cycle']:.1f}%")
    print(f"    Jaccard similarity: {overlap['jaccard_similarity']:.4f}")

    conn.close()


if __name__ == "__main__":
    main()