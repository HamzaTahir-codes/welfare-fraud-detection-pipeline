"""
evaluate_isolation_forest.py

Stage 7 - Isolation Forest evaluation against planted ground truth
--------------------------------------------------------------------

Primary answer key: ground truth rows with fraud_type == "multivariate_anomaly"
(beneficiary-level). Evaluated at the BENEFICIARY level: fraud_signals.entity_id
is a beneficiaries.row_id, mapped back to beneficiary_id.

Why more than one number? Isolation Forest is unsupervised and looks for
"unusual", not for one specific planted pattern. Other planted frauds are also
unusual (e.g. shared-account victims are paid ~11 times because they receive
both their own and hub-account payments). Those flags are not really wrong, but
the multivariate_anomaly answer key does not list them. So this report shows:

  1. Official result: latest stored isolation_forest run in fraud_signals,
     P/R/F1 vs multivariate_anomaly, and a breakdown of WHAT the flagged set
     contains (planted anomaly / shared-account victim / duplicate identity /
     unlabeled).
  2. Contamination sweep: same scores, different flagging cut-offs. Scores do
     not depend on contamination, so this recomputes the model once (same seed).
  3. Ranking quality (ROC-AUC, average precision), on the whole scored
     population AND on "planted anomalies vs ordinary beneficiaries only".
  4. Incremental value: the same ranking on the beneficiaries that the
     VERIFIED Stage 5/6 detectors (shared_bank_account, fuzzy duplicate
     identity) do NOT already cover -- i.e. what Isolation Forest adds.
  5. Reference baseline: a one-line rule "n_payments > MAX" -- NOT a Stage 7
     detector, included so the comparison is honest (see note in output).

"Unlabeled" flags are either novel findings or legitimate rare behaviour --
the ground truth cannot tell us which. Do not call them false alarms or wins.

Usage:
    python -m src.fraud_detection.evaluate_isolation_forest
    python -m src.fraud_detection.evaluate_isolation_forest --ground-truth data/ground_truth_fraud_ids.csv
    python -m src.fraud_detection.evaluate_isolation_forest --layered   # after isolation_forest.py --layered
"""

import argparse

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.database.connection import get_connection
from src.fraud_detection.isolation_forest import (
    DEFAULT_N_ESTIMATORS,
    DEFAULT_SEED,
    METHOD,
    SIGNAL_TYPE,
    build_features,
    exclude_covered,
    fit_and_score,
    load_covered_ids,
    load_inputs,
)

SWEEP_CONTAMINATIONS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]


def load_ground_truth(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def build_answer_keys(gt: pd.DataFrame) -> dict:
    mv = set(gt.loc[gt["fraud_type"] == "multivariate_anomaly", "beneficiary_id"].dropna())
    shared = set(gt.loc[gt["fraud_type"] == "shared_account", "beneficiary_id"].dropna())
    dup = gt[gt["fraud_type"] == "duplicate_identity"]
    dup_ids = set(dup["original_beneficiary_id"].dropna()) | set(dup["duplicate_beneficiary_id"].dropna())
    return {"multivariate_anomaly": mv, "shared_account_victim": shared, "duplicate_identity": dup_ids}


def load_latest_run_signals(conn) -> pd.DataFrame:
    query = f"""
        SELECT * FROM fraud_signals
        WHERE method = '{METHOD}'
          AND run_id = (
              SELECT run_id FROM fraud_signals
              WHERE method = '{METHOD}'
              ORDER BY created_at DESC
              LIMIT 1
          );
    """
    df = pd.read_sql(query, conn)
    df["entity_id"] = df["entity_id"].astype(str)
    return df


def precision_recall_f1(predicted: set, actual: set) -> dict:
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def composition(flagged_ids: set, keys: dict) -> dict:
    """What is inside the flagged set? Categories can overlap, so 'unlabeled' = in none of them."""
    labeled = set().union(*keys.values())
    return {
        "planted multivariate_anomaly": len(flagged_ids & keys["multivariate_anomaly"]),
        "shared-account victims": len(flagged_ids & keys["shared_account_victim"]),
        "duplicate-identity beneficiaries": len(flagged_ids & keys["duplicate_identity"]),
        "unlabeled (novel or legitimate-rare)": len(flagged_ids - labeled),
    }


def print_report(name: str, m: dict):
    print(f"\n[{name}]")
    print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(f"    Precision: {m['precision']:.4f}")
    print(f"    Recall:    {m['recall']:.4f}")
    print(f"    F1:        {m['f1']:.4f}")


def evaluate_official(signals: pd.DataFrame, df_ben: pd.DataFrame, keys: dict, scored_ids: set):
    row_to_ben = dict(zip(df_ben["row_id"].astype(str), df_ben["beneficiary_id"]))
    flagged_ids = {
        row_to_ben[e] for e in signals.loc[signals["signal_type"] == SIGNAL_TYPE, "entity_id"] if e in row_to_ben
    }

    # Recall denominator = planted anomalies that were actually scoreable (>=1 payment).
    actual = keys["multivariate_anomaly"] & scored_ids
    print(f"Planted multivariate anomalies: {len(keys['multivariate_anomaly'])} "
          f"({len(actual)} scoreable, i.e. have at least one payment)")
    print_report("isolation_forest vs multivariate_anomaly (beneficiary-level)",
                 precision_recall_f1(flagged_ids, actual))

    print("\n[what the flagged set contains]  (flagged: %d)" % len(flagged_ids))
    for label, n in composition(flagged_ids, keys).items():
        print(f"    {label:<40s} {n}")

    print("\n[per planted type: how many did Isolation Forest flag?]")
    for label, ids in keys.items():
        scoreable = ids & scored_ids
        hit = len(flagged_ids & scoreable)
        pct = hit / len(scoreable) * 100 if scoreable else 0.0
        print(f"    {label:<24s} {hit}/{len(scoreable)} scoreable  ({pct:.1f}%)")


def evaluate_sweep_and_ranking(scored: pd.DataFrame, keys: dict):
    scored = scored.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    ids = scored["beneficiary_id"]
    mv = keys["multivariate_anomaly"]
    other = keys["shared_account_victim"] | keys["duplicate_identity"]
    actual = mv & set(ids)

    print("\n[contamination sweep -- top-k% of scores flagged]")
    print(f"    {'contam':>7} {'flagged':>8} {'TP':>4} {'prec':>7} {'recall':>7} {'F1':>7} "
          f"{'| shared':>9} {'dup-id':>7} {'unlabeled':>10}")
    for c in SWEEP_CONTAMINATIONS:
        k = int(round(len(scored) * c))
        flagged_ids = set(ids.iloc[:k])
        m = precision_recall_f1(flagged_ids, actual)
        comp = composition(flagged_ids, keys)
        print(f"    {c:>7.3f} {k:>8d} {m['tp']:>4d} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} "
              f"{'| ' + str(comp['shared-account victims']):>9} {comp['duplicate-identity beneficiaries']:>7d} "
              f"{comp['unlabeled (novel or legitimate-rare)']:>10d}")

    y_all = ids.isin(mv).astype(int)
    print("\n[ranking quality -- independent of any threshold]")
    print(f"    whole scored population       ROC-AUC={roc_auc_score(y_all, scored['anomaly_score']):.4f}  "
          f"AP={average_precision_score(y_all, scored['anomaly_score']):.4f}  "
          f"(random-guess AP = {y_all.mean():.4f})")

    clean = scored[~ids.isin(other - mv)]
    y_clean = clean["beneficiary_id"].isin(mv).astype(int)
    print(f"    planted anomalies vs ordinary  ROC-AUC={roc_auc_score(y_clean, clean['anomaly_score']):.4f}  "
          f"AP={average_precision_score(y_clean, clean['anomaly_score']):.4f}  "
          f"(other planted frauds removed; n={len(clean)})")


BASELINE_MAX_PAYMENTS = 6   # simulation window is 6 monthly cycles -> more than 6 payments is impossible normally
TOP_K_LIST = [10, 25, 50, 100]


def evaluate_incremental(scored: pd.DataFrame, keys: dict, covered_ids: set):
    mv = keys["multivariate_anomaly"]
    resid = scored[~scored["beneficiary_id"].isin(covered_ids)].sort_values(
        "anomaly_score", ascending=False).reset_index(drop=True)
    y = resid["beneficiary_id"].isin(mv).astype(int)
    n_pos = int(y.sum())

    print("\n[incremental value -- beyond what verified Stage 5/6 detectors already cover]")
    if covered_ids:
        print(f"    Beneficiaries covered by shared_bank_account / fuzzy duplicate: {len(covered_ids & set(scored['beneficiary_id']))}")
    print(f"    Residual population: {len(resid)}  (planted anomalies inside it: {n_pos}/{len(mv)})")
    if n_pos == 0 or n_pos == len(resid):
        print("    (not enough positives/negatives to score)")
        return
    print(f"    ROC-AUC={roc_auc_score(y, resid['anomaly_score']):.4f}  "
          f"AP={average_precision_score(y, resid['anomaly_score']):.4f}  (random-guess AP = {y.mean():.4f})")
    for k in TOP_K_LIST:
        tp = int(y.iloc[:k].sum())
        print(f"    top-{k:<4d} of ranking: {tp:>3d} planted anomalies  "
              f"(precision {tp / k:.2f}, recall {tp / n_pos:.2f})")

    base_flag = resid["n_payments"] > BASELINE_MAX_PAYMENTS
    tp = int((base_flag & (y == 1)).sum())
    print(f"\n[reference baseline on the same residual population: n_payments > {BASELINE_MAX_PAYMENTS}]")
    print(f"    flagged={int(base_flag.sum())}  TP={tp}/{n_pos}  "
          f"precision={tp / max(int(base_flag.sum()), 1):.3f}  recall={tp / n_pos:.3f}")
    print("    NOTE: the planted pattern gives every anomalous profile exactly 7 payments while no legitimate")
    print("    beneficiary can exceed 6, so a single hard-coded threshold separates them. Isolation Forest is")
    print("    never told that feature or ceiling; the baseline shows what an analyst who already knew where to")
    print("    look would achieve. Report both.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Isolation Forest against planted ground truth.")
    parser.add_argument("--ground-truth", type=str, default="data/ground_truth_fraud_ids.csv")
    parser.add_argument("--layered", action="store_true",
                        help="Evaluate a run made with isolation_forest.py --layered (population excludes "
                             "beneficiaries already flagged by Stage 5/6).")
    args = parser.parse_args()

    conn = get_connection()
    try:
        gt = load_ground_truth(args.ground_truth)
        keys = build_answer_keys(gt)
        signals = load_latest_run_signals(conn)
        df_ben, df_disb = load_inputs(conn)
        covered_ids = load_covered_ids(conn)
    finally:
        conn.close()

    if signals.empty:
        print("No isolation_forest signals found in fraud_signals. Run isolation_forest.py first (without --dry-run).")
        return

    feats = build_features(df_ben, df_disb, verbose=False)
    if args.layered:
        feats = exclude_covered(feats, covered_ids)
    scored = fit_and_score(feats, contamination=0.05, n_estimators=DEFAULT_N_ESTIMATORS, seed=DEFAULT_SEED)
    scored_ids = set(scored["beneficiary_id"])

    print(f"Evaluating run_id: {signals['run_id'].iloc[0]}")
    print(f"Ground truth entries: {len(gt)} ({', '.join(f'{k}={len(v)}' for k, v in keys.items())})")

    evaluate_official(signals, df_ben, keys, scored_ids)
    evaluate_sweep_and_ranking(scored, keys)
    evaluate_incremental(scored, keys, set() if args.layered else covered_ids)
    print("\nNote: sweep/ranking recompute scores with the default seed and hyperparameters; "
          "if you ran isolation_forest.py with a different --seed/--n-estimators, they will differ slightly "
          "from the stored run.")


if __name__ == "__main__":
    main()