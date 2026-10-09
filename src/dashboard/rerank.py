"""
rerank.py -- re-rank an EXISTING dashboard snapshot offline (no database).

Rebuilds case_queue.csv of one snapshot from that snapshot's own files, using
the same ranking function publish.py uses (ranking.build_case_queue):
- signals_long.csv                          -> the evidence per beneficiary
- results/comparison_detector_metrics.csv   -> the per-signal weights
- case_queue.csv (the old one)              -> which ids are in the registry

Then it refreshes the queue-derived fields of kpis.json (plus the weights
used), records the re-rank in manifest.json and, unless --no-report, rebuilds
report.html through report.build_report, which also copies it to the
gitignored data/reports/fraud_report_<snapshot>.html and
data/reports/fraud_report_latest.html. The answer key is never read.

Running it twice gives a byte-identical case_queue.csv and kpis.json, but
manifest.json (reranked_at) and report.html (which embeds the manifest) get
a new timestamp on every run, so regenerate and commit those together.

Usage (from project root):
    python -m src.dashboard.rerank --snapshot 20260922052809 --dry-run
    python -m src.dashboard.rerank --snapshot 20260922052809
    python -m src.dashboard.rerank                  # snapshot latest.json points at
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.dashboard.ranking import build_case_queue, queue_kpis, signal_weights, weight_map, weights_records

SNAPSHOT_ROOT = Path("data") / "dashboard"


def _write_atomic(path: Path, text: str) -> None:
    """Temp file + rename, so the dashboard never reads a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def rerank_frames(
    signals_long: pd.DataFrame, old_queue: pd.DataFrame, metrics: Optional[pd.DataFrame], **weight_options
):
    """Pure core of the re-rank: (new queue, weights table) from in-memory frames.

    weight_options go to ranking.signal_weights (priors=, precision_column=);
    leave them out for the shipped ranking.
    """
    # The registry is not in the snapshot, but the old queue recorded membership per id.
    in_reg = old_queue["in_registry"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    registry_ids = set(old_queue.loc[in_reg, "beneficiary_id"])

    weights = signal_weights(metrics, **weight_options)
    queue = build_case_queue(signals_long, registry_ids, weight_map(weights))

    # Same beneficiaries as before, only the order (and the new score column) may change.
    if set(queue["beneficiary_id"]) != set(old_queue["beneficiary_id"]):
        raise ValueError("Re-ranked queue does not contain the same beneficiaries as the old case_queue.csv")
    return queue, weights


def rerank_snapshot(snapshot_dir, dry_run: bool = False) -> pd.DataFrame:
    """Re-rank one snapshot folder in place. Returns the new queue."""
    d = Path(snapshot_dir)
    signals_long = pd.read_csv(d / "signals_long.csv")
    old_queue = pd.read_csv(d / "case_queue.csv")
    metrics_path = d / "results" / "comparison_detector_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
    if metrics is None:
        print(f"[rerank] WARNING: {metrics_path} not found; every signal gets the heuristic prior.")

    queue, weights = rerank_frames(signals_long, old_queue, metrics)

    print("[rerank] signal weights:")
    print(weights.to_string(index=False))
    print("\n[rerank] top 10:")
    print(queue.head(10).to_string(index=False))
    if dry_run:
        print("\n[rerank] Dry run -- nothing written.")
        return queue

    _write_atomic(d / "case_queue.csv", queue.to_csv(index=False))

    kpis = json.loads((d / "kpis.json").read_text())
    kpis.update(queue_kpis(queue))
    kpis["ranking_weights"] = weights_records(weights)
    _write_atomic(d / "kpis.json", json.dumps(kpis, indent=2, default=str))

    manifest = json.loads((d / "manifest.json").read_text())
    manifest["reranked_at"] = datetime.now(timezone.utc).isoformat()
    manifest["queue_size"] = int(len(queue))
    _write_atomic(d / "manifest.json", json.dumps(manifest, indent=2, default=str))
    print(f"\n[rerank] Wrote case_queue.csv, kpis.json and manifest.json in {d}")
    return queue


def main():
    parser = argparse.ArgumentParser(description="Re-rank an existing dashboard snapshot offline.")
    parser.add_argument("--snapshot", default=None, help="Snapshot id. Default: the one latest.json points at.")
    parser.add_argument("--dry-run", action="store_true", help="Print the new ranking without writing anything.")
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not rebuild report.html (by default it is rebuilt and also copied to data/reports/).",
    )
    args = parser.parse_args()

    snap = args.snapshot or json.loads((SNAPSHOT_ROOT / "latest.json").read_text())["snapshot_id"]
    snapshot_dir = SNAPSHOT_ROOT / snap
    rerank_snapshot(snapshot_dir, dry_run=args.dry_run)

    if not args.dry_run and not args.no_report:
        from src.dashboard.report import build_report

        print(f"[rerank] Report rebuilt: {build_report(snapshot_dir)}")


if __name__ == "__main__":
    main()
