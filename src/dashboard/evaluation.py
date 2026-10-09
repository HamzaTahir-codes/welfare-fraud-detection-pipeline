"""
evaluation.py -- grade the investigation queue against the planted answer key.

EVALUATION ONLY. Nothing in here may ever feed the ranking: ranking.py,
publish.py and rerank.py must not import this module. It exists so a human
(and the Investigation queue page) can see how much real planted fraud sits
at the top of the queue.

A beneficiary counts as "planted" if its id appears in ANY of the columns
beneficiary_id / original_beneficiary_id / duplicate_beneficiary_id of
data/ground_truth_fraud_ids.csv (both members of a planted duplicate pair are
part of the fraud).

Usage (from project root):
    python -m src.dashboard.evaluation                          # latest snapshot
    python -m src.dashboard.evaluation --snapshot 20260922052809
    python -m src.dashboard.evaluation --queue-csv path/to/case_queue.csv
    python -m src.dashboard.evaluation --legacy       # the same queue in the OLD order
    python -m src.dashboard.evaluation --sensitivity  # old order + several weightings

--legacy and --sensitivity only re-order in memory; they write nothing.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ANSWER_KEY = Path("data") / "ground_truth_fraud_ids.csv"
SNAPSHOT_ROOT = Path("data") / "dashboard"
ID_COLUMNS = ["beneficiary_id", "original_beneficiary_id", "duplicate_beneficiary_id"]
DEFAULT_KS = (25, 50, 100, 250, 500)


def planted_ids(answer_key: pd.DataFrame) -> set:
    """Every beneficiary id named anywhere in the answer key."""
    ids = set()
    for col in ID_COLUMNS:
        if col in answer_key.columns:
            ids |= set(answer_key[col].dropna().astype(str).str.strip())
    ids.discard("")
    return ids


def load_planted_ids(path=ANSWER_KEY) -> Optional[set]:
    """Planted ids from the answer key file, or None if the file is missing."""
    path = Path(path)
    if not path.exists():
        return None
    return planted_ids(pd.read_csv(path, dtype=str))


def precision_at_k(queue: pd.DataFrame, planted: set, ks: Iterable[int] = DEFAULT_KS) -> pd.DataFrame:
    """Share of the top-k queue entries (by 'rank') that are planted fraud.

    Columns: k, n_planted, precision. k larger than the queue is capped at
    the queue length so the share is never computed over missing rows.
    """
    ordered = queue.sort_values("rank")["beneficiary_id"].astype(str).tolist()
    hits = [b in planted for b in ordered]
    rows = []
    for k in ks:
        top = hits[: min(int(k), len(hits))]
        n = sum(top)
        rows.append({"k": int(k), "n_planted": int(n), "precision": n / len(top) if top else float("nan")})
    return pd.DataFrame(rows, columns=["k", "n_planted", "precision"])


# Below this share of planted ids present in the queue, the answer key most
# likely belongs to a different generator run than the snapshot being graded.
MIN_ANSWER_KEY_COVERAGE = 0.5


def answer_key_coverage(queue: pd.DataFrame, planted: set) -> float:
    """Share of planted ids that appear anywhere in the queue (NaN if none are planted).

    A sanity check, not a metric: the answer key in data/ is the one for the
    current generator run, so grading a snapshot from another run gives a low
    coverage (and meaningless precision@k).
    """
    if not planted:
        return float("nan")
    return len(planted & set(queue["beneficiary_id"].astype(str))) / len(planted)


def legacy_order(queue: pd.DataFrame) -> pd.DataFrame:
    """The queue in the ranking used before ranking.py: validated families,
    then all families, then highest score, then id. Used to show 'before'."""
    out = queue.sort_values(
        ["n_validated_families", "n_families", "max_score", "beneficiary_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out


# (label, keyword arguments for ranking.signal_weights). The first row is
# what ships; the others show how much each in-sample choice matters.
SENSITIVITY_CONFIGS = [
    ("shipped: per-signal column, cnic_mismatch prior 0.01", {}),
    ("per-signal column, flat prior 0.02", {"priors": {}}),
    ("precision_strict everywhere, cnic_mismatch prior 0.01", {"precision_column": "precision_strict"}),
    ("precision_strict everywhere, flat prior 0.02", {"precision_column": "precision_strict", "priors": {}}),
    ("plain precision everywhere, cnic_mismatch prior 0.01", {"precision_column": "precision"}),
    ("plain precision everywhere, flat prior 0.02", {"precision_column": "precision", "priors": {}}),
]


def sensitivity_table(snapshot_dir, planted: set, ks: Iterable[int] = DEFAULT_KS) -> pd.DataFrame:
    """precision@k of the old order and of each SENSITIVITY_CONFIGS weighting,
    rebuilt in memory from the snapshot's own files. One row per config."""
    from src.dashboard.rerank import rerank_frames  # the import goes this way only

    d = Path(snapshot_dir)
    signals_long = pd.read_csv(d / "signals_long.csv")
    queue = pd.read_csv(d / "case_queue.csv")
    metrics_path = d / "results" / "comparison_detector_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None

    rows = []
    candidates = [("before: legacy order (validated families, families, score)", legacy_order(queue))]
    for label, options in SENSITIVITY_CONFIGS:
        candidates.append((label, rerank_frames(signals_long, queue, metrics, **options)[0]))
    for label, q in candidates:
        t = precision_at_k(q, planted, ks)
        rows.append({"ranking": label, **{f"@{k}": p for k, p in zip(t["k"], t["precision"])}})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Precision@k of the investigation queue (evaluation only).")
    parser.add_argument("--snapshot", default=None, help="Snapshot id. Default: the one latest.json points at.")
    parser.add_argument("--queue-csv", default=None, help="Grade this case_queue.csv instead of a snapshot.")
    parser.add_argument("--answer-key", default=str(ANSWER_KEY))
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS) + [1000])
    parser.add_argument("--legacy", action="store_true", help="Grade the queue re-sorted by the old ranking rule.")
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Grade the old order and several weightings, rebuilt in memory from a snapshot. Writes nothing.",
    )
    args = parser.parse_args()

    snap = args.snapshot or json.loads((SNAPSHOT_ROOT / "latest.json").read_text())["snapshot_id"]
    queue_path = Path(args.queue_csv) if args.queue_csv else SNAPSHOT_ROOT / snap / "case_queue.csv"

    planted = load_planted_ids(args.answer_key)
    if planted is None:
        raise SystemExit(f"Answer key not found: {args.answer_key}")

    if args.sensitivity:
        table = sensitivity_table(SNAPSHOT_ROOT / snap, planted, args.k)
        print(f"[evaluation] snapshot {snap} re-ranked in memory, graded against {args.answer_key}")
        pct = {c: table[c].map(lambda p: f"{p:.1%}") for c in table.columns if c.startswith("@")}
        print(table.assign(**pct).to_string(index=False))
        return

    queue = pd.read_csv(queue_path)
    if args.legacy:
        queue = legacy_order(queue)
    table = precision_at_k(queue, planted, args.k)
    order = " (re-sorted by the legacy rule)" if args.legacy else ""
    print(f"[evaluation] {queue_path}{order} graded against {args.answer_key} ({len(planted)} planted ids)")
    coverage = answer_key_coverage(queue, planted)
    print(f"[evaluation] {coverage:.1%} of the planted ids appear in this queue")
    if coverage < MIN_ANSWER_KEY_COVERAGE:
        print("[evaluation] WARNING: the answer key probably belongs to a different run than this queue.")
    print(table.assign(precision=table["precision"].map(lambda p: f"{p:.1%}")).to_string(index=False))


if __name__ == "__main__":
    main()
