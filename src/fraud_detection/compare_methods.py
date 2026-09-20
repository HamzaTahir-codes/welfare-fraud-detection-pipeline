"""
compare_methods.py

Stage 8 - Comparative Evaluation of the Three Fraud-Detection Methods
----------------------------------------------------------------------

Compares Stage 5 (rule-based), Stage 6 (fuzzy record linkage) and Stage 7
(Isolation Forest) against the SAME planted answer key
(data/ground_truth_fraud_ids.csv), on a common unit of evaluation.

WHY A COMMON UNIT
    Each method emits a different thing: rule-based -> disbursement/beneficiary
    row_ids in fraud_signals, fuzzy -> beneficiary PAIRS in
    record_linkage_matches, Isolation Forest -> beneficiary row_ids in
    fraud_signals. Everything is mapped to the business `beneficiary_id`
    (including fabricated IDs such as BEN-900000, which is what makes orphan
    disbursements comparable). Set operations on that key drive every number.

GROUND-TRUTH ROLES (per planted pattern)
    core      = the planted culprit(s): shared_account victims, duplicate
                identities, fabricated orphan IDs, planted anomalies.
    involved  = core + counterparts that are genuinely part of the fraud but
                were never logged as culprits:
                  shared_account      -> + the hub-account OWNER
                  duplicate_identity  -> + the ORIGINAL beneficiary
                This resolves the known "hub owner logged as false positive"
                gap (see evaluate_rule_based.py) in one consistent way.

METRICS
    recall             = |flagged & core[target]| / |core[target]|
    precision          = |flagged & involved[target]| / |flagged|
    precision_strict   = |flagged & core[target]| / |flagged|   (tie-out with
                         the earlier per-stage evaluators)
    any_fraud_precision= |flagged & involved[any pattern]| / |flagged|
                         ("is this a real planted fraud of ANY kind?")
    Detectors with no planted ground truth (the four heuristic rules) get
    NO target metrics -- only counts and any_fraud_precision, reported as
    informative, never as validated.

This module is READ-ONLY against Postgres and the answer key (flag-never-drop).

Usage:
    python -m src.fraud_detection.compare_methods --list-runs
    python -m src.fraud_detection.compare_methods
    python -m src.fraud_detection.compare_methods --if-run-id <uuid> --if-layered-run-id <uuid>
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────
GROUND_TRUTH_PATH = Path("data") / "ground_truth_fraud_ids.csv"
RESULTS_DIR = Path("data") / "results"

RULE_METHOD = "rule_based"
IF_METHOD_PREFIX = "isolation_forest"   # ASSUMPTION: matches your Stage 7 `method` value(s)
FUZZY_MATCH_METHOD = "fuzzy_duplicate_identity"

# Simulated program window is 6 monthly cycles -> more distinct payments than
# this cannot happen for a beneficiary paid once per cycle.
MAX_CYCLES_IN_WINDOW = 6

PATTERNS = ["orphan_disbursement", "shared_account", "duplicate_identity", "multivariate_anomaly"]

# rule signal_type -> planted pattern it is validated against
VALIDATED_RULE_SIGNALS = {
    "orphaned_disbursement": "orphan_disbursement",
    "shared_bank_account": "shared_account",
}
UNVALIDATED_RULE_SIGNALS = [
    "duplicate_cnic_multiple_identities",
    "cnic_mismatch",
    "duplicate_cycle_payment",
    "payment_to_inactive_beneficiary",
]


# ── DATA STRUCTURES ─────────────────────────────────────────────────────

@dataclass
class Truth:
    core: Dict[str, Set[str]] = field(default_factory=dict)
    involved: Dict[str, Set[str]] = field(default_factory=dict)

    @property
    def all_core(self) -> Set[str]:
        return set().union(*self.core.values()) if self.core else set()

    @property
    def all_involved(self) -> Set[str]:
        return set().union(*self.involved.values()) if self.involved else set()


@dataclass
class Detector:
    name: str
    family: str                  # rule_based | fuzzy | isolation_forest | baseline | ensemble
    level: str                   # "signal" (one detector) or "method" (method-level rollup)
    flagged: Set[str]            # business beneficiary_ids
    target: Optional[str] = None # planted pattern it is validated against (single-target only)
    validated: bool = False


# ── GROUND TRUTH -> ENTITY SETS ─────────────────────────────────────────

def build_truth(gt: pd.DataFrame, beneficiaries: pd.DataFrame) -> Truth:
    truth = Truth()

    def rows(fraud_type: str) -> pd.DataFrame:
        return gt[gt["fraud_type"] == fraud_type]

    orphan = set(rows("orphan_disbursement")["beneficiary_id"].dropna())
    truth.core["orphan_disbursement"] = orphan
    truth.involved["orphan_disbursement"] = set(orphan)

    shared = rows("shared_account")
    victims = set(shared["beneficiary_id"].dropna())
    hub_accounts = set(shared["bank_account_number"].dropna())
    hub_owners = set(
        beneficiaries.loc[beneficiaries["bank_account_number"].isin(hub_accounts), "beneficiary_id"].dropna()
    )
    truth.core["shared_account"] = victims
    truth.involved["shared_account"] = victims | hub_owners

    dup = rows("duplicate_identity")
    duplicates = set(dup["duplicate_beneficiary_id"].dropna())
    originals = set(dup["original_beneficiary_id"].dropna())
    truth.core["duplicate_identity"] = duplicates
    truth.involved["duplicate_identity"] = duplicates | originals

    anomalies = rows("multivariate_anomaly")
    if len(anomalies):
        # ASSUMPTION: generate_anomalous_profiles.py logs the planted beneficiary
        # in the `beneficiary_id` column, like shared_account / orphan rows do.
        if "beneficiary_id" not in anomalies.columns or anomalies["beneficiary_id"].isna().all():
            raise ValueError(
                "multivariate_anomaly rows found in ground truth but `beneficiary_id` is empty/missing. "
                "Adjust build_truth() to the column generate_anomalous_profiles.py actually writes."
            )
        ids = set(anomalies["beneficiary_id"].dropna())
    else:
        print("[WARN] No multivariate_anomaly rows in the ground-truth file -- Stage 7 pattern will be "
              "empty. Are you pointing at the ground truth written AFTER generate_anomalous_profiles.py?")
        ids = set()
    truth.core["multivariate_anomaly"] = ids
    truth.involved["multivariate_anomaly"] = set(ids)

    return truth


# ── SIGNALS -> BUSINESS beneficiary_id ──────────────────────────────────

def build_row_maps(beneficiaries: pd.DataFrame, disbursements: pd.DataFrame):
    ben_map = dict(zip(beneficiaries["row_id"].astype(str), beneficiaries["beneficiary_id"]))
    disb_map = dict(zip(disbursements["row_id"].astype(str), disbursements["beneficiary_id"]))
    known_ids = set(beneficiaries["beneficiary_id"].dropna()) | set(disbursements["beneficiary_id"].dropna())
    return ben_map, disb_map, known_ids


def signals_to_beneficiaries(sig: pd.DataFrame, ben_map: dict, disb_map: dict, known_ids: set, label: str) -> Set[str]:
    flagged, unmapped = set(), 0
    for etype, eid in zip(sig["entity_type"], sig["entity_id"].astype(str)):
        bid = (ben_map if etype == "beneficiary" else disb_map).get(eid)
        if bid is None and eid in known_ids:   # tolerate a stage that stored the business ID directly
            bid = eid
        if bid is None or pd.isna(bid):
            unmapped += 1
            continue
        flagged.add(bid)
    if unmapped:
        print(f"[WARN] {label}: {unmapped} signal rows could not be mapped to a beneficiary_id "
              f"(stale row_ids? re-run Stage 5/7 after the last loader_stage run).")
    return flagged


def select_run(signals: pd.DataFrame, method_filter, run_id: Optional[str], label: str) -> pd.DataFrame:
    sub = signals[method_filter(signals["method"])]
    if sub.empty:
        print(f"[WARN] {label}: no rows in fraud_signals for this method.")
        return sub
    if run_id:
        chosen = sub[sub["run_id"] == run_id]
        if chosen.empty:
            raise ValueError(f"{label}: run_id {run_id} not found. Use --list-runs.")
        return chosen
    latest = sub.sort_values("created_at").iloc[-1]["run_id"]
    print(f"[INFO] {label}: using latest run_id {latest} ({(sub['run_id'] == latest).sum()} signal rows)")
    return sub[sub["run_id"] == latest]


# ── DETECTOR CONSTRUCTION ───────────────────────────────────────────────

def payment_count_baseline(disbursements: pd.DataFrame) -> Set[str]:
    """Reference baseline, NOT a proposed detector: more distinct payments than
    there are cycles in the program window. Distinct disbursement_id collapses
    inject_messiness.py's exact-copy rows."""
    valid = disbursements[disbursements["beneficiary_id"].notna() & disbursements["disbursement_id"].notna()]
    counts = valid.groupby("beneficiary_id")["disbursement_id"].nunique()
    return set(counts[counts > MAX_CYCLES_IN_WINDOW].index)


def build_detectors(
    signals: pd.DataFrame,
    rlm: pd.DataFrame,
    beneficiaries: pd.DataFrame,
    disbursements: pd.DataFrame,
    if_run_id: Optional[str] = None,
    if_layered_run_id: Optional[str] = None,
) -> List[Detector]:
    ben_map, disb_map, known_ids = build_row_maps(beneficiaries, disbursements)
    dets: List[Detector] = []

    # ---- Stage 5: rule-based (latest run only) ----
    rb = select_run(signals, lambda m: m == RULE_METHOD, None, "rule_based")
    rule_sets: Dict[str, Set[str]] = {}
    for sig_type, grp in rb.groupby("signal_type"):
        rule_sets[sig_type] = signals_to_beneficiaries(grp, ben_map, disb_map, known_ids, f"rule:{sig_type}")

    for sig_type, pattern in VALIDATED_RULE_SIGNALS.items():
        dets.append(Detector(f"rule:{sig_type}", "rule_based", "signal",
                             rule_sets.get(sig_type, set()), target=pattern, validated=True))
    for sig_type in UNVALIDATED_RULE_SIGNALS:
        dets.append(Detector(f"rule:{sig_type}", "rule_based", "signal",
                             rule_sets.get(sig_type, set()), target=None, validated=False))

    # ---- Stage 6: fuzzy linkage (flagged pairs -> both members) ----
    fz = rlm[(rlm["match_method"] == FUZZY_MATCH_METHOD) & (rlm["is_flagged"] == True)]  # noqa: E712
    fuzzy_flagged = set(fz["beneficiary_id"].dropna()) | set(fz["matched_record_id"].dropna())
    dets.append(Detector("fuzzy:duplicate_identity", "fuzzy", "signal", fuzzy_flagged,
                         target="duplicate_identity", validated=True))

    # ---- Stage 7: Isolation Forest ----
    is_if = lambda m: m.str.startswith(IF_METHOD_PREFIX)
    if_sig = select_run(signals, is_if, if_run_id, "isolation_forest")
    if_flagged = signals_to_beneficiaries(if_sig, ben_map, disb_map, known_ids, "isolation_forest") if len(if_sig) else set()
    dets.append(Detector("iforest:standard", "isolation_forest", "signal", if_flagged,
                         target="multivariate_anomaly", validated=True))
    if if_layered_run_id:
        lay_sig = select_run(signals, is_if, if_layered_run_id, "isolation_forest (layered)")
        lay_flagged = signals_to_beneficiaries(lay_sig, ben_map, disb_map, known_ids, "isolation_forest (layered)")
        dets.append(Detector("iforest:layered", "isolation_forest", "signal", lay_flagged,
                             target="multivariate_anomaly", validated=True))

    # ---- Reference baseline ----
    dets.append(Detector("baseline:payment_count_gt_6", "baseline", "signal",
                         payment_count_baseline(disbursements), target="multivariate_anomaly", validated=False))

    # ---- Method-level rollups (what the presentation compares) ----
    by_name = {d.name: d for d in dets}
    rb_valid = set().union(*[by_name[f"rule:{s}"].flagged for s in VALIDATED_RULE_SIGNALS])
    rb_all = set().union(*[d.flagged for d in dets if d.family == "rule_based"])
    dets.append(Detector("METHOD rule_based (validated 2)", "rule_based", "method", rb_valid))
    dets.append(Detector("METHOD rule_based (all 6)", "rule_based", "method", rb_all))
    dets.append(Detector("METHOD fuzzy linkage", "fuzzy", "method", fuzzy_flagged))
    dets.append(Detector("METHOD isolation_forest (standard)", "isolation_forest", "method", if_flagged))
    best_if = if_flagged
    if if_layered_run_id:
        best_if = by_name["iforest:layered"].flagged
        dets.append(Detector("METHOD isolation_forest (layered)", "isolation_forest", "method", best_if))
    dets.append(Detector("METHOD ENSEMBLE (rules2 + fuzzy + IF)", "ensemble", "method",
                         rb_valid | fuzzy_flagged | best_if))
    return dets


# ── SCORING ─────────────────────────────────────────────────────────────

def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def score_detectors(dets: List[Detector], truth: Truth) -> pd.DataFrame:
    rows = []
    any_inv = truth.all_involved
    all_core = truth.all_core
    for d in dets:
        n = len(d.flagged)
        row = {
            "detector": d.name, "family": d.family, "level": d.level,
            "has_ground_truth": d.target is not None, "target": d.target or "-",
            "n_flagged": n,
            "any_fraud_precision": _safe_div(len(d.flagged & any_inv), n),
            "overall_core_recall": _safe_div(len(d.flagged & all_core), len(all_core)),
        }
        core = truth.core.get(d.target, set()) if d.target else set()
        if d.target and core:
            inv = truth.involved[d.target]
            tp_core, tp_inv = len(d.flagged & core), len(d.flagged & inv)
            precision, recall = _safe_div(tp_inv, n), _safe_div(tp_core, len(core))
            f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
            row.update({"tp_core": tp_core, "core_size": len(core), "precision": precision,
                        "recall": recall, "f1": f1, "precision_strict": _safe_div(tp_core, n)})
        else:
            row.update({"tp_core": np.nan, "core_size": np.nan, "precision": np.nan,
                        "recall": np.nan, "f1": np.nan, "precision_strict": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_matrix(dets: List[Detector], truth: Truth, level: str = "method") -> pd.DataFrame:
    """rows = planted pattern, cols = detector, value = recall on that pattern's core entities."""
    sel = [d for d in dets if d.level == level]
    data = {}
    for d in sel:
        data[d.name] = [_safe_div(len(d.flagged & truth.core[p]), len(truth.core[p])) for p in PATTERNS]
    m = pd.DataFrame(data, index=PATTERNS)
    m.index.name = "planted_pattern"
    return m


def unique_catch_table(dets: List[Detector], truth: Truth) -> pd.DataFrame:
    """Among the three methods (+ rule split), how many planted core entities does each catch that
    NO other method catches? This is the honest case for (or against) running all three."""
    methods = [d for d in dets if d.name in (
        "METHOD rule_based (validated 2)", "METHOD fuzzy linkage",
        "METHOD isolation_forest (layered)", "METHOD isolation_forest (standard)")]
    # prefer layered IF if present, otherwise standard
    if any(d.name == "METHOD isolation_forest (layered)" for d in methods):
        methods = [d for d in methods if d.name != "METHOD isolation_forest (standard)"]
    core_all = truth.all_core
    rows = []
    for d in methods:
        others = set().union(*[o.flagged for o in methods if o is not d])
        caught = d.flagged & core_all
        rows.append({"method": d.name, "core_caught": len(caught),
                     "caught_by_this_method_only": len(caught - others),
                     "core_total": len(core_all)})
    return pd.DataFrame(rows)


def overlap_matrix(dets: List[Detector]) -> pd.DataFrame:
    sel = [d for d in dets if d.level == "method" and d.family != "ensemble"]
    sel.append(next(d for d in dets if d.family == "baseline"))
    names = [d.name for d in sel]
    m = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in sel:
        for b in sel:
            union = a.flagged | b.flagged
            m.loc[a.name, b.name] = _safe_div(len(a.flagged & b.flagged), len(union))
    m.index.name = "jaccard"
    return m


def fuzzy_pair_tieout(rlm: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """Recompute Stage 6's pair-level precision/recall so this module can be checked against it."""
    dup = gt[gt["fraud_type"] == "duplicate_identity"]
    actual = {frozenset({a, b}) for a, b in zip(dup["original_beneficiary_id"], dup["duplicate_beneficiary_id"])}
    fz = rlm[(rlm["match_method"] == FUZZY_MATCH_METHOD) & (rlm["is_flagged"] == True)]  # noqa: E712
    predicted = {frozenset({a, b}) for a, b in zip(fz["beneficiary_id"], fz["matched_record_id"])}
    tp = len(predicted & actual)
    p, r = _safe_div(tp, len(predicted)), _safe_div(tp, len(actual))
    return {"pairs_flagged": len(predicted), "pairs_true": len(actual), "tp": tp, "precision": p, "recall": r}


# ── REPORTING ───────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is pd.NA or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "n/a"
    if isinstance(v, (float, np.floating)):
        return f"{v:.3f}"
    return str(v)


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def write_report(path: Path, scores: pd.DataFrame, cov: pd.DataFrame, uniq: pd.DataFrame,
                 overlap: pd.DataFrame, tie: dict, truth: Truth) -> None:
    sizes = pd.DataFrame({"pattern": PATTERNS,
                          "core_entities": [len(truth.core[p]) for p in PATTERNS],
                          "involved_entities": [len(truth.involved[p]) for p in PATTERNS]})
    method_cols = ["detector", "n_flagged", "any_fraud_precision", "overall_core_recall"]
    signal_cols = ["detector", "has_ground_truth", "target", "n_flagged", "tp_core", "core_size",
                   "precision", "recall", "f1", "precision_strict", "any_fraud_precision"]
    lines = [
        "# Stage 8 - Comparative Evaluation of Fraud-Detection Methods", "",
        "All numbers are measured against the PLANTED answer key. Perfect or near-perfect scores mean "
        "'catches the specific planted pattern', not 'works on real welfare fraud'.", "",
        "## Answer key size", md_table(sizes), "",
        "## Coverage matrix - recall on each planted pattern, by method",
        md_table(cov, index=True), "",
        "## Method scorecard",
        md_table(scores[scores["level"] == "method"][method_cols]), "",
        "## Signal-level detail (P/R/F1 against each detector's own target pattern)",
        md_table(scores[scores["level"] == "signal"][signal_cols].astype({"tp_core": "Int64", "core_size": "Int64"})), "",
        "## What each method catches that the others miss (core planted entities)",
        md_table(uniq), "",
        "## Flagged-set overlap (Jaccard)", md_table(overlap, index=True), "",
        "## Tie-out with Stage 6 (pair-level)", md_table(pd.DataFrame([tie])), "",
        "## How to read this honestly",
        "- `precision` counts hub-account owners (shared_account) and original beneficiaries (duplicate_identity) "
        "as legitimately involved; `precision_strict` counts only the logged culprits and is the number "
        "comparable with the earlier per-stage evaluators.",
        "- `any_fraud_precision` credits a flag if the entity is part of ANY planted fraud. A detector with low "
        "target precision but high any-fraud precision is finding real fraud of a different kind.",
        "- The four heuristic rules have no planted ground truth: their counts are informative, not validated. "
        "`cnic_mismatch` in particular fires on generator noise (national ID CNIC typos), so a low "
        "any-fraud precision there is not evidence the rule is wrong.",
        "- `baseline:payment_count_gt_6` is a sanity reference. If a one-line count threshold matches "
        "Isolation Forest on the planted anomalies, that reflects how the anomalies were planted "
        "(an extra out-of-window payment), not necessarily that Isolation Forest is weak in general.",
        "- The layered Isolation Forest excludes beneficiaries already flagged by Stage 5/6 before fitting, "
        "so it is designed to be complementary, not a like-for-like rival.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_coverage(cov: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib not installed - skipping heatmap.")
        return
    data = cov.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.9 * data.shape[1] + 2, 0.9 * data.shape[0] + 2))
    im = ax.imshow(np.nan_to_num(data, nan=0.0), vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([c.replace("METHOD ", "").replace(" (", "\n(") for c in cov.columns], fontsize=8)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(cov.index, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, "n/a" if np.isnan(data[i, j]) else f"{data[i, j]:.2f}",
                    ha="center", va="center", fontsize=9)
    ax.set_title("Recall on planted fraud patterns, by method")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ── ORCHESTRATION ───────────────────────────────────────────────────────

def run_comparison(gt, beneficiaries, disbursements, signals, rlm,
                   if_run_id=None, if_layered_run_id=None) -> dict:
    truth = build_truth(gt, beneficiaries)
    dets = build_detectors(signals, rlm, beneficiaries, disbursements, if_run_id, if_layered_run_id)
    return {
        "truth": truth,
        "detectors": dets,
        "scores": score_detectors(dets, truth),
        "coverage": coverage_matrix(dets, truth),
        "unique": unique_catch_table(dets, truth),
        "overlap": overlap_matrix(dets),
        "tieout": fuzzy_pair_tieout(rlm, gt),
    }


def load_db_tables(conn):
    beneficiaries = pd.read_sql("SELECT row_id, beneficiary_id, bank_account_number FROM beneficiaries;", conn)
    disbursements = pd.read_sql("SELECT row_id, disbursement_id, beneficiary_id FROM disbursements;", conn)
    signals = pd.read_sql(
        "SELECT entity_id, entity_type, method, signal_type, score, run_id, created_at FROM fraud_signals;", conn)
    signals["entity_id"] = signals["entity_id"].astype(str)
    rlm = pd.read_sql(
        "SELECT beneficiary_id, matched_record_id, match_method, confidence_score, is_flagged "
        "FROM record_linkage_matches;", conn)
    return beneficiaries, disbursements, signals, rlm


def print_runs(signals: pd.DataFrame) -> None:
    g = (signals.groupby(["method", "run_id"])
         .agg(rows=("entity_id", "size"), first_created=("created_at", "min"),
              signal_types=("signal_type", lambda s: ", ".join(sorted(set(s)))))
         .reset_index().sort_values("first_created"))
    print(g.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Stage 8: comparative evaluation of the three detection methods.")
    parser.add_argument("--ground-truth", type=str, default=str(GROUND_TRUTH_PATH))
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--if-run-id", type=str, default=None, help="Isolation Forest (standard) run_id; default = latest.")
    parser.add_argument("--if-layered-run-id", type=str, default=None, help="Isolation Forest --layered run_id (optional).")
    parser.add_argument("--list-runs", action="store_true", help="Print method/run_id inventory of fraud_signals and exit.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    from src.database.connection import get_connection  # same import path as rule_based.py
    conn = get_connection()
    try:
        beneficiaries, disbursements, signals, rlm = load_db_tables(conn)
    finally:
        conn.close()

    if args.list_runs:
        print_runs(signals)
        return

    gt = pd.read_csv(args.ground_truth, dtype=str)
    result = run_comparison(gt, beneficiaries, disbursements, signals, rlm,
                            args.if_run_id, args.if_layered_run_id)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    result["scores"].to_csv(out / "comparison_detector_metrics.csv", index=False)
    result["coverage"].to_csv(out / "comparison_coverage_matrix.csv")
    result["unique"].to_csv(out / "comparison_unique_catches.csv", index=False)
    result["overlap"].to_csv(out / "comparison_overlap_jaccard.csv")
    write_report(out / "comparison_report.md", result["scores"], result["coverage"], result["unique"],
                 result["overlap"], result["tieout"], result["truth"])
    if not args.no_plot:
        plot_coverage(result["coverage"], out / "comparison_coverage_heatmap.png")

    print("\n=== Coverage matrix (recall on planted patterns) ===")
    print(result["coverage"].round(3).to_string())
    print("\n=== Method scorecard ===")
    sc = result["scores"]
    print(sc[sc["level"] == "method"][["detector", "n_flagged", "any_fraud_precision", "overall_core_recall"]]
          .round(3).to_string(index=False))
    print(f"\nStage 6 pair-level tie-out: {result['tieout']}")
    print(f"\nWrote results to {out.resolve()}")


if __name__ == "__main__":
    main()