"""
publish.py -- Stage 10a: build the dashboard snapshot.

Reads the finished detector results, combines them into one ranked
investigation queue (one row per beneficiary_id), and saves a snapshot
folder that the Streamlit app reads. The app never touches Postgres, so a
DAG run that truncates the tables can never blank the dashboard.

Usage (from project root):
    python -m src.dashboard.publish --if-layered-run-id <uuid> --dry-run
    python -m src.dashboard.publish --if-layered-run-id <uuid>
    python -m src.dashboard.publish --if-layered-run-id <uuid> --dag-run-id <id>

Get the layered run id with:
    python -m src.fraud_detection.compare_methods --list-runs

Design rules this file respects:
- Flag, never drop: the queue is built from what detectors flagged.
- Ground truth (ground_truth_fraud_ids.csv) is NEVER read here. Use it only
  to test the output by hand.
- Signals are grouped by evidence family; scores are NOT summed.
"""

import argparse
import json
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.database.connection import get_connection  # same import as rule_based.py

# ── CONFIG ────────────────────────────────────────────────────────────────
SNAPSHOT_ROOT = Path("data") / "dashboard"
RESULTS_DIR = Path("data") / "results"

# signal_type -> evidence family. Correlated signals share a family so they
# count as ONE piece of evidence (shared_bank_account and
# duplicate_cycle_payment are not independent).
FAMILY_MAP = {
    "duplicate_cnic_multiple_identities": "identity",
    "cnic_mismatch": "identity",
    "fuzzy_duplicate_identity": "identity",
    "shared_bank_account": "payment_channel",
    "duplicate_cycle_payment": "payment_channel",
    "orphaned_disbursement": "existence",
    "payment_to_inactive_beneficiary": "eligibility",
    "multivariate_anomaly": "statistical",
}

# Signals validated against planted ground truth. Used only as a tie-break
# so an unvalidated heuristic alone never outranks a validated method.
VALIDATED_SIGNALS = {
    "orphaned_disbursement",
    "shared_bank_account",
    "fuzzy_duplicate_identity",
    "multivariate_anomaly",   # layered run only
}

KPI_TABLES = [
    "beneficiaries", "national_id_records", "disbursements",
    "fraud_signals", "record_linkage_matches",
]

BENEFICIARY_DQ_COLS = ["income_flag", "phone_missing", "address_missing", "is_duplicate_row"]
DISBURSEMENT_DQ_COLS = ["amount_flag", "payment_method_missing", "disb_date_parse_failed"]


# ── 1. RUN IDS ────────────────────────────────────────────────────────────

def get_run_ids(conn, if_layered_run_id: str) -> dict:
    """Latest rule_based run + the layered Isolation Forest run you pass in."""
    latest = pd.read_sql(
        """
        SELECT run_id FROM fraud_signals
        WHERE method = 'rule_based'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        conn,
    )
    if latest.empty:
        raise ValueError("No rule_based signals in fraud_signals. Run rule_based.py first.")
    rule_run_id = str(latest["run_id"].iloc[0])

    check = pd.read_sql(
        "SELECT COUNT(*) AS n FROM fraud_signals WHERE method = 'isolation_forest' AND run_id = %s",
        conn,
        params=(if_layered_run_id,),
    )
    if int(check["n"].iloc[0]) == 0:
        raise ValueError(
            f"No isolation_forest signals found for run_id {if_layered_run_id}. "
            "Check the id with: python -m src.fraud_detection.compare_methods --list-runs"
        )

    return {"rule_based": rule_run_id, "isolation_forest_layered": str(if_layered_run_id)}


# ── 2. SIGNALS -> ONE ROW PER (beneficiary, method, signal_type) ─────────

def build_signals_long(conn, run_ids: dict) -> pd.DataFrame:
    """Columns: beneficiary_id, method, signal_type, family, score, n_flags."""
    # row_id -> beneficiary_id lookups (fraud_signals.entity_id is text)
    ben_map = pd.read_sql("SELECT row_id, beneficiary_id FROM beneficiaries", conn)
    disb_map = pd.read_sql("SELECT row_id, beneficiary_id FROM disbursements", conn)
    ben_lookup = ben_map.assign(row_id=ben_map["row_id"].astype(str)).set_index("row_id")["beneficiary_id"]
    disb_lookup = disb_map.assign(row_id=disb_map["row_id"].astype(str)).set_index("row_id")["beneficiary_id"]

    signals = pd.read_sql(
        """
        SELECT entity_id, entity_type, method, signal_type, score
        FROM fraud_signals
        WHERE (method = 'rule_based' AND run_id = %s)
           OR (method = 'isolation_forest' AND run_id = %s)
        """,
        conn,
        params=(run_ids["rule_based"], run_ids["isolation_forest_layered"]),
    )
    signals["entity_id"] = signals["entity_id"].astype(str)
    signals["score"] = pd.to_numeric(signals["score"])

    signals["beneficiary_id"] = pd.Series(index=signals.index, dtype="object")
    is_ben = signals["entity_type"] == "beneficiary"
    is_disb = signals["entity_type"] == "disbursement"
    signals.loc[is_ben, "beneficiary_id"] = signals.loc[is_ben, "entity_id"].map(ben_lookup)
    signals.loc[is_disb, "beneficiary_id"] = signals.loc[is_disb, "entity_id"].map(disb_lookup)

    n_unmapped = int(signals["beneficiary_id"].isna().sum())
    print(f"[publish] fraud_signals rows loaded: {len(signals)}; could not map to a beneficiary_id: {n_unmapped}")
    signals = signals[signals["beneficiary_id"].notna()]

    # Fuzzy pairs -> two rows each (both members are flagged)
    pairs = pd.read_sql(
        """
        SELECT beneficiary_id, matched_record_id, confidence_score
        FROM record_linkage_matches
        WHERE is_flagged = TRUE AND match_method = 'fuzzy_duplicate_identity'
        """,
        conn,
    )
    side_a = pairs[["beneficiary_id", "confidence_score"]].rename(columns={"confidence_score": "score"})
    side_b = pairs[["matched_record_id", "confidence_score"]].rename(
        columns={"matched_record_id": "beneficiary_id", "confidence_score": "score"}
    )
    fuzzy = pd.concat([side_a, side_b], ignore_index=True)
    fuzzy["score"] = pd.to_numeric(fuzzy["score"])
    fuzzy["method"] = "fuzzy_linkage"
    fuzzy["signal_type"] = "fuzzy_duplicate_identity"
    print(f"[publish] fuzzy pairs flagged: {len(pairs)} -> {len(fuzzy)} beneficiary rows")

    cols = ["beneficiary_id", "method", "signal_type", "score"]
    combined = pd.concat([signals[cols], fuzzy[cols]], ignore_index=True)

    combined["family"] = combined["signal_type"].map(FAMILY_MAP)
    unknown = sorted(combined.loc[combined["family"].isna(), "signal_type"].unique())
    if unknown:
        raise ValueError(f"signal_type(s) missing from FAMILY_MAP: {unknown}")

    # Collapse to one row per (beneficiary, method, signal_type). This also
    # collapses the duplicate rows inject_messiness.py made (same
    # beneficiary_id, different row_id).
    long_df = (
        combined.groupby(["beneficiary_id", "method", "signal_type", "family"], as_index=False)
        .agg(score=("score", "max"), n_flags=("score", "size"))
    )
    return long_df


# ── 3. THE RANKED QUEUE ──────────────────────────────────────────────────

def build_case_queue(signals_long: pd.DataFrame, registry_ids: set) -> pd.DataFrame:
    """One row per beneficiary_id, most suspicious first.

    Rank order: number of distinct VALIDATED evidence families, then number of
    all distinct families, then highest score, then id (deterministic).

    Why validated families first: cnic_mismatch mostly comes from the typos
    the generator injects on purpose, and payment_to_inactive_beneficiary
    co-occurs with shared-account victims because those victims were sampled
    from ALL beneficiaries, not only Active ones. Two unvalidated heuristics
    stacked on one person are weaker evidence than they look.
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
        df[df["is_validated"]].groupby("beneficiary_id")["family"].nunique()
        .rename("n_validated_families")
    )
    queue = queue.join(validated_families, on="beneficiary_id")
    queue["n_validated_families"] = queue["n_validated_families"].fillna(0).astype(int)

    queue = queue.sort_values(
        ["n_validated_families", "n_families", "max_score", "beneficiary_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    queue.insert(0, "rank", range(1, len(queue) + 1))
    return queue[[
        "rank", "beneficiary_id", "n_validated_families", "n_families", "families",
        "methods", "signal_types", "max_score", "has_validated_signal",
        "total_flags", "in_registry",
    ]]


# ── 4. HEADLINE NUMBERS ─────────────────────────────────────────────────

def _count_rows(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")  # table names come from KPI_TABLES only
        return int(cur.fetchone()[0])


def _flag_rates(conn, table: str, cols: list) -> dict:
    df = pd.read_sql(f"SELECT {', '.join(cols)} FROM {table}", conn)
    return {c: round(float(df[c].fillna(False).astype(bool).mean()), 4) for c in cols}


def build_kpis(conn, signals_long: pd.DataFrame, case_queue: pd.DataFrame) -> dict:
    by_type = (
        signals_long.groupby(["method", "signal_type"])["beneficiary_id"]
        .nunique()
        .reset_index(name="n_beneficiaries")
    )
    return {
        "row_counts": {t: _count_rows(conn, t) for t in KPI_TABLES},
        "dq_flag_rates": {
            "beneficiaries": _flag_rates(conn, "beneficiaries", BENEFICIARY_DQ_COLS),
            "disbursements": _flag_rates(conn, "disbursements", DISBURSEMENT_DQ_COLS),
        },
        "signals_by_type": by_type.to_dict("records"),
        "queue_size": int(len(case_queue)),
        "queue_by_families": {
            int(k): int(v) for k, v in case_queue["n_families"].value_counts().sort_index().items()
        },
        "not_in_registry": int((~case_queue["in_registry"]).sum()),
    }


# ── 5. SAVE THE SNAPSHOT ────────────────────────────────────────────────

def write_snapshot(
    case_queue: pd.DataFrame,
    signals_long: pd.DataFrame,
    kpis: dict,
    run_ids: dict,
    dag_run_id: Optional[str] = None,
) -> Path:
    """Write data/dashboard/<snapshot_id>/ then swap latest.json LAST.

    The pointer is written to a temp file and moved with os.replace, which is
    atomic: the dashboard sees either the old snapshot or the new one, never a
    half-written run. Same "fixed-name latest" idea as Stage 3.
    """
    now = datetime.now(timezone.utc)   # same clock on your Mac and in the Airflow container
    snapshot_id = now.strftime("%Y%m%d%H%M%S")
    snapshot_dir = SNAPSHOT_ROOT / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    case_queue.to_csv(snapshot_dir / "case_queue.csv", index=False)
    signals_long.to_csv(snapshot_dir / "signals_long.csv", index=False)

    # Copy the Stage 8 comparison files so the snapshot is self-contained.
    results_dir = snapshot_dir / "results"
    results_dir.mkdir()
    copied = []
    for f in sorted(RESULTS_DIR.glob("comparison_*")):
        shutil.copy2(f, results_dir / f.name)
        copied.append(f.name)

    manifest = {
        "snapshot_id": snapshot_id,
        "published_at": now.isoformat(),
        "dag_run_id": dag_run_id,
        "run_ids": run_ids,
        "queue_size": int(len(case_queue)),
        "signals_long_rows": int(len(signals_long)),
        "comparison_files_copied": copied,
    }
    (snapshot_dir / "kpis.json").write_text(json.dumps(kpis, indent=2, default=str))
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # Pointer LAST, atomically.
    tmp = SNAPSHOT_ROOT / "latest.json.tmp"
    tmp.write_text(json.dumps({
        "snapshot_id": snapshot_id,
        "path": snapshot_id,
        "published_at": now.isoformat(),
    }, indent=2))
    os.replace(tmp, SNAPSHOT_ROOT / "latest.json")

    # Read it back: proves the pointer really moved (this shows up in the Airflow task log).
    now_points_at = json.loads((SNAPSHOT_ROOT / "latest.json").read_text())["snapshot_id"]
    if now_points_at != snapshot_id:
        raise RuntimeError(f"latest.json still points at {now_points_at}, expected {snapshot_id}")
    print(f"[publish] latest.json verified: points at {now_points_at}")

    return snapshot_dir


# ── MAIN ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build the dashboard snapshot (Stage 10a).")
    parser.add_argument("--if-layered-run-id", required=True,
                        help="run_id of the layered Isolation Forest run.")
    parser.add_argument("--dag-run-id", default=None,
                        help="Airflow run id, recorded in the manifest.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and print the queue without writing a snapshot.")
    args = parser.parse_args()

    conn = get_connection()
    try:
        run_ids = get_run_ids(conn, args.if_layered_run_id)
        signals_long = build_signals_long(conn, run_ids)
        registry_ids = set(
            pd.read_sql("SELECT DISTINCT beneficiary_id FROM beneficiaries", conn)["beneficiary_id"]
        )
        case_queue = build_case_queue(signals_long, registry_ids)
        kpis = build_kpis(conn, signals_long, case_queue)
    finally:
        conn.close()

    print(f"[publish] run ids: {run_ids}")
    print(f"[publish] signals_long rows: {len(signals_long)}")
    print(f"[publish] case_queue rows:   {len(case_queue)}")
    print("[publish] queue size by number of families flagged:")
    print(case_queue["n_families"].value_counts().sort_index().to_string())
    print("\n[publish] top 10:")
    print(case_queue.head(10).to_string(index=False))

    if args.dry_run:
        print("\n[publish] Dry run -- no snapshot written.")
        return

    snapshot_dir = write_snapshot(case_queue, signals_long, kpis, run_ids, args.dag_run_id)
    print(f"\n[publish] Snapshot written to {snapshot_dir} and latest.json updated.")

    # The written report is built from the snapshot just published. A report failure must
    # never invalidate a good snapshot, so it only warns (the app can rebuild it on demand).
    try:
        from src.dashboard.report import build_report
        report_path = build_report(snapshot_dir)
        print(f"[publish] Report written to {report_path}")
    except Exception as exc:  # noqa: BLE001
        print("[publish] " + "!" * 60)
        print(f"[publish] WARNING: snapshot is published but the REPORT STEP FAILED: {exc}")
        traceback.print_exc()
        print("[publish] " + "!" * 60)


if __name__ == "__main__":
    main()