"""
supervised.py

Stage 7b - Supervised baseline next to the unsupervised Isolation Forest (offline)
-----------------------------------------------------------------------------------

Question: if we were ALLOWED to learn from the planted answer key, how much
better than Isolation Forest (and than a one-line rule) could a model do on
the same per-beneficiary features? This is a reference point, not a detector:
nothing is written to the database and nothing here feeds the dashboard.

Runs fully offline from tracked files (no Postgres, no .env):
    data/clean/beneficiaries_cleaned.csv
    data/clean/disbursements_cleaned.csv
    data/ground_truth_fraud_ids.csv

Features: exactly isolation_forest.build_features (imported, not copied), so
the definitions cannot drift:
    n_payments, mean_amount, total_amount, income, amount_to_income
The CSVs have no database row_id, so file order (1-based) stands in for it;
build_features only uses row_id to pick one row per duplicated beneficiary_id.

Labels (from the answer key, attached AFTER features are built):
    multivariate_anomaly  primary target, beneficiary_id rows of that type
                          (25 positives -> every fold has ~5; read the spread)
    any_planted           any beneficiary named anywhere in the answer key
                          (beneficiary_id, original_beneficiary_id,
                          duplicate_beneficiary_id), any fraud_type

Compared on the SAME stratified folds (RepeatedStratifiedKFold, fixed seed):
    hgb               HistGradientBoostingClassifier (supervised)
    logreg            StandardScaler + LogisticRegression (supervised, linear)
    isolation_forest  project params (300 trees, contamination 0.05, seed 42),
                      fit on the training fold WITHOUT labels, scores the test fold
    rule_n_payments_gt_6
                      score = 1 if n_payments > 6 else 0 (no training)

Feature sets (the honesty check):
    all                   the five project features
    no_count              drops n_payments ... but total_amount = mean_amount *
                          n_payments still encodes it exactly
    no_count_no_total     drops n_payments AND total_amount -- what is left is
                          amount and income only

Metrics per test fold: PR-AUC (average precision), ROC-AUC, and precision /
recall when flagging the top 1% and top 5% of the fold. Ties at the budget
cut-off are split in expectation (a 0/1 rule gets its random-tie-break value,
not a lucky ordering). Summaries are mean / std / min / max over all folds.

Leakage guard: assert_no_leakage() refuses any feature column that is, or is
derived from, the answer key; tests/test_supervised.py pins it.

Output (deterministic, no timestamps):
    data/results/supervised_metrics.csv
    data/results/supervised_report.md

Usage:
    python -m src.fraud_detection.supervised
    python -m src.fraud_detection.supervised --n-splits 5 --n-repeats 5 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.fraud_detection.isolation_forest import (
    DEFAULT_CONTAMINATION,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_SEED,
    FEATURE_COLUMNS,
    build_features,
)

BENEFICIARIES_PATH = Path("data") / "clean" / "beneficiaries_cleaned.csv"
DISBURSEMENTS_PATH = Path("data") / "clean" / "disbursements_cleaned.csv"
GROUND_TRUTH_PATH = Path("data") / "ground_truth_fraud_ids.csv"
RESULTS_DIR = Path("data") / "results"
METRICS_FILE = "supervised_metrics.csv"
REPORT_FILE = "supervised_report.md"

PRIMARY_TARGET = "multivariate_anomaly"
SECONDARY_TARGET = "any_planted"
TARGETS = [PRIMARY_TARGET, SECONDARY_TARGET]
LABEL_PREFIX = "y_"

FEATURE_SETS = {
    "all": list(FEATURE_COLUMNS),
    "no_count": [c for c in FEATURE_COLUMNS if c != "n_payments"],
    "no_count_no_total": [c for c in FEATURE_COLUMNS if c not in ("n_payments", "total_amount")],
}

RULE_MAX_PAYMENTS = 6  # same reference rule as evaluate_isolation_forest.py
RULE_METHOD = f"rule_n_payments_gt_{RULE_MAX_PAYMENTS}"
RULE_FEATURE_SET = "n_payments_only"
MODEL_METHODS = ["hgb", "logreg", "isolation_forest"]

BUDGETS = [0.01, 0.05]  # share of each test fold that gets flagged
METRIC_ORDER = ["pr_auc", "roc_auc"] + [f"{m}_at_{int(b * 100)}pct" for b in BUDGETS for m in ("precision", "recall")]

DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 5
PERM_REPEATS = 10

# Column names that come from (or are built from) the answer key. None may ever be a feature.
GROUND_TRUTH_COLUMNS = {
    "fraud_type",
    "bank_account_number",
    "beneficiary_id",
    "original_beneficiary_id",
    "duplicate_beneficiary_id",
    "run_id",
}
# Identifiers: not answer-key columns, but planted records have recognisable IDs
# (e.g. duplicates were appended as BEN-015001+), so row order / IDs leak too.
IDENTIFIER_COLUMNS = {"row_id", "beneficiary_id"}


# ---------------------------------------------------------------------
# Data (pure functions where possible)
# ---------------------------------------------------------------------


def load_accounts(ben_path=BENEFICIARIES_PATH) -> pd.DataFrame:
    """beneficiary_id -> bank_account_number, only for the report's hub-owner count."""
    return pd.read_csv(ben_path, dtype=str, usecols=["beneficiary_id", "bank_account_number"])


def load_clean_inputs(ben_path=BENEFICIARIES_PATH, disb_path=DISBURSEMENTS_PATH):
    """Read the tracked cleaned CSVs into the shape isolation_forest.load_inputs returns."""
    df_ben = pd.read_csv(ben_path, dtype=str, usecols=["beneficiary_id", "monthly_income_clean"])
    df_disb = pd.read_csv(disb_path, dtype=str, usecols=["disbursement_id", "beneficiary_id", "amount_pkr_clean"])
    return add_row_ids(df_ben), df_disb


def add_row_ids(df_ben: pd.DataFrame) -> pd.DataFrame:
    """CSV stand-in for the DB serial row_id: 1-based file order (what the loader inserts in)."""
    out = df_ben.reset_index(drop=True).copy()
    out.insert(0, "row_id", np.arange(1, len(out) + 1))
    return out


def answer_key_sets(gt: pd.DataFrame) -> dict:
    """beneficiary_id sets per target. Only ever used for labels, never for features."""
    mv = set(gt.loc[gt["fraud_type"] == "multivariate_anomaly", "beneficiary_id"].dropna())
    named = set()
    for col in ("beneficiary_id", "original_beneficiary_id", "duplicate_beneficiary_id"):
        if col in gt.columns:
            named |= set(gt[col].dropna())
    return {PRIMARY_TARGET: mv, SECONDARY_TARGET: named}


def answer_key_roles(gt: pd.DataFrame, ben_accounts: pd.DataFrame = None) -> dict:
    """{fraud_type: beneficiary_ids named in any id column of that type's rows}, plus
    'hub_owner': account holders of planted shared accounts, as compare_methods.build_truth
    counts them in `involved`. Report context only, never features."""
    roles = {}
    for fraud_type, rows in gt.groupby("fraud_type"):
        ids = set()
        for col in ("beneficiary_id", "original_beneficiary_id", "duplicate_beneficiary_id"):
            if col in rows.columns:
                ids |= set(rows[col].dropna())
        roles[fraud_type] = ids
    if ben_accounts is not None and "bank_account_number" in gt.columns:
        hubs = set(gt.loc[gt["fraud_type"] == "shared_account", "bank_account_number"].dropna())
        roles["hub_owner"] = set(
            ben_accounts.loc[ben_accounts["bank_account_number"].isin(hubs), "beneficiary_id"].dropna()
        )
    return roles


def role_breakdown(ids, roles: dict) -> str:
    """e.g. '570 shared_account (1 of them also duplicate_identity)': planted ids by fraud type."""
    ids = set(ids)
    types = [t for t in sorted(roles) if t != "hub_owner"]
    hits = {t: ids & roles[t] for t in types if ids & roles[t]}
    if not hits:
        return "none of them in the answer key"
    # each id is counted once, under its largest type; overlaps are noted, not double counted
    parts, counted = [], set()
    for t, members in sorted(hits.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        new = members - counted
        if not new:
            continue
        counted |= new
        also = {o: len(new & roles[o]) for o in types if o != t and new & roles[o]}
        extra = "; ".join(f"{n} of them also {o}" for o, n in also.items())
        parts.append(f"{len(new)} {t}" + (f" ({extra})" if extra else ""))
    return ", ".join(parts)


def budget_ceiling(n_positives: int, n: int, budget: float, n_splits: int = 1) -> float:
    """Highest recall any ranking can reach at `budget`, computed like fold_metrics does: each of
    n_splits stratified test folds flags round(budget * fold size) rows and holds ~1/n_splits of
    the positives."""
    k = round(budget * n / n_splits)
    return min(1.0, k / max(n_positives / n_splits, 1e-9))


def attach_labels(feats: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of feats with one 0/1 `y_<target>` column per target."""
    out = feats.copy()
    for target, ids in answer_key_sets(gt).items():
        out[LABEL_PREFIX + target] = out["beneficiary_id"].isin(ids).astype(int)
    return out


def assert_no_leakage(feature_columns, extra_forbidden=()) -> None:
    """Fail loudly if any feature is an answer-key column, a label, an identifier,
    or anything outside the project's feature definitions."""
    cols = list(feature_columns)
    forbidden = GROUND_TRUTH_COLUMNS | IDENTIFIER_COLUMNS | set(extra_forbidden)
    bad = [c for c in cols if c in forbidden or c.startswith(LABEL_PREFIX)]
    if bad:
        raise ValueError(f"Leakage guard: answer-key/label/identifier columns used as features: {bad}")
    unknown = [c for c in cols if c not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"Leakage guard: features outside isolation_forest.FEATURE_COLUMNS: {unknown}")


# ---------------------------------------------------------------------
# Scoring and metrics
# ---------------------------------------------------------------------


def make_model(method: str, seed: int):
    if method == "hgb":
        return HistGradientBoostingClassifier(random_state=seed)
    if method == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    raise ValueError(f"unknown supervised method {method}")


def score_fold(method: str, X_train, y_train, X_test, seed: int):
    """Higher score = more suspicious. Returns (test scores, fitted model or None)."""
    if method == "isolation_forest":
        # Unsupervised: y_train is deliberately NOT passed.
        model = IsolationForest(
            n_estimators=DEFAULT_N_ESTIMATORS,
            max_samples="auto",
            contamination=DEFAULT_CONTAMINATION,
            random_state=DEFAULT_SEED,
            n_jobs=-1,
        )
        model.fit(X_train)
        return -model.score_samples(X_test), None
    model = make_model(method, seed)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1], model


def rule_scores(n_payments) -> np.ndarray:
    return (np.asarray(n_payments, dtype=float) > RULE_MAX_PAYMENTS).astype(float)


def expected_hits_at_k(y, scores, k: int) -> float:
    """Positives among the top-k by score; ties straddling the cut-off are split in expectation."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    k = int(min(max(k, 0), len(scores)))
    if k == 0:
        return 0.0
    cutoff = np.sort(scores)[::-1][k - 1]
    above = scores > cutoff
    tied = scores == cutoff
    n_take = k - int(above.sum())
    return float(y[above].sum() + y[tied].sum() * n_take / tied.sum())


def fold_metrics(y, scores) -> dict:
    y = np.asarray(y)
    out = {"pr_auc": average_precision_score(y, scores), "roc_auc": roc_auc_score(y, scores)}
    n_pos = int(y.sum())
    for b in BUDGETS:
        k = int(round(b * len(y)))
        hits = expected_hits_at_k(y, scores, k)
        out[f"precision_at_{int(b * 100)}pct"] = hits / k if k else np.nan
        out[f"recall_at_{int(b * 100)}pct"] = hits / n_pos if n_pos else np.nan
    return out


# ---------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------


def run_cv(data: pd.DataFrame, target: str, n_splits: int, n_repeats: int, seed: int, verbose: bool = True):
    """One row per (feature_set, method, repeat, fold) with fold metrics, plus
    permutation importances for hgb on every fold (scored by average precision)."""
    y = data[LABEL_PREFIX + target].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows, importances = [], []
    for fold_idx, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        repeat, fold = divmod(fold_idx, n_splits)
        base = {"target": target, "repeat": repeat, "fold": fold, "n_test": len(te), "n_test_pos": int(y[te].sum())}

        rule = rule_scores(data["n_payments"].to_numpy()[te])
        rows.append({**base, "feature_set": RULE_FEATURE_SET, "method": RULE_METHOD, **fold_metrics(y[te], rule)})

        for fs_name, cols in FEATURE_SETS.items():
            assert_no_leakage(cols)
            X = data[cols].to_numpy(dtype=float)
            for method in MODEL_METHODS:
                scores, model = score_fold(method, X[tr], y[tr], X[te], seed)
                rows.append({**base, "feature_set": fs_name, "method": method, **fold_metrics(y[te], scores)})
                if method == "hgb":
                    pi = permutation_importance(
                        model,
                        X[te],
                        y[te],
                        scoring="average_precision",
                        n_repeats=PERM_REPEATS,
                        random_state=seed,
                    )
                    for col, val in zip(cols, pi.importances_mean):
                        importances.append({"target": target, "feature_set": fs_name, "feature": col, "value": val})
        if verbose and fold == n_splits - 1:
            print(f"[supervised] {target}: repeat {repeat + 1}/{n_repeats} done")
    return pd.DataFrame(rows), pd.DataFrame(importances)


def summarise(folds: pd.DataFrame, importances: pd.DataFrame) -> pd.DataFrame:
    """Long, tidy summary: one row per (target, feature_set, method, metric)."""
    long = folds.melt(
        id_vars=["target", "feature_set", "method", "repeat", "fold"],
        value_vars=METRIC_ORDER,
        var_name="metric",
    )
    keys = ["target", "feature_set", "method", "metric"]
    cv = long.groupby(keys, sort=False)["value"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    cv.insert(0, "kind", "cv_metric")

    imp = importances.rename(columns={"feature": "metric"}).copy()
    imp["method"] = "hgb"
    imp = imp.groupby(keys, sort=False)["value"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    imp["metric"] = "perm_importance_ap:" + imp["metric"]
    imp.insert(0, "kind", "permutation_importance")

    out = pd.concat([cv, imp], ignore_index=True).rename(columns={"count": "n_folds"})
    for col in ["mean", "std", "min", "max"]:
        out[col] = out[col].astype(float).round(4)
    return out


def rule_operating_point(data: pd.DataFrame, target: str) -> dict:
    """The rule on the whole population (no folds): how many it flags and what it catches."""
    y = data[LABEL_PREFIX + target].to_numpy()
    flag = rule_scores(data["n_payments"]) == 1
    tp = int((flag & (y == 1)).sum())
    return {
        "flagged": int(flag.sum()),
        "tp": tp,
        "positives": int(y.sum()),
        "precision": tp / max(int(flag.sum()), 1),
        "recall": tp / max(int(y.sum()), 1),
    }


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------


def summary_row(summary: pd.DataFrame, target, feature_set, method, metric, kind="cv_metric"):
    """The single summary row for one cell, or None."""
    row = summary[
        (summary["kind"] == kind)
        & (summary["target"] == target)
        & (summary["feature_set"] == feature_set)
        & (summary["method"] == method)
        & (summary["metric"] == metric)
    ]
    return None if row.empty else row.iloc[0]


def _fmt(summary: pd.DataFrame, target: str, feature_set: str, method: str, metric: str) -> str:
    r = summary_row(summary, target, feature_set, method, metric)
    if r is None:
        return "n/a"
    return f"{r['mean']:.3f} ± {r['std']:.3f} [{r['min']:.2f}-{r['max']:.2f}]"


def metric_table(summary: pd.DataFrame, target: str) -> str:
    header = "| feature set | method | " + " | ".join(METRIC_ORDER) + " |"
    lines = [header, "|" + "---|" * (len(METRIC_ORDER) + 2)]
    combos = [(RULE_FEATURE_SET, RULE_METHOD)] + [(fs, m) for fs in FEATURE_SETS for m in MODEL_METHODS]
    for fs, m in combos:
        cells = [_fmt(summary, target, fs, m, metric) for metric in METRIC_ORDER]
        lines.append(f"| {fs} | {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def importance_table(summary: pd.DataFrame, target: str) -> str:
    imp = summary[(summary["kind"] == "permutation_importance") & (summary["target"] == target)]
    lines = ["| feature set | feature | mean drop in AP | std | min | max |", "|---|---|---|---|---|---|"]
    for fs in FEATURE_SETS:
        sub = imp[imp["feature_set"] == fs].sort_values("mean", ascending=False, kind="mergesort")
        for _, r in sub.iterrows():
            feat = r["metric"].split(":", 1)[1]
            lines.append(f"| {fs} | {feat} | {r['mean']:.4f} | {r['std']:.4f} | {r['min']:.4f} | {r['max']:.4f} |")
    return "\n".join(lines)


def mean_of(summary, target, fs, method, metric, kind="cv_metric") -> float:
    return float(summary_row(summary, target, fs, method, metric, kind)["mean"])


# The report prints means with 3 decimals; every comparison word is decided at that
# same precision, so the text cannot contradict the table it sits next to.
REPORT_DIGITS = 3


def relation(a: float, b: float, digits: int = REPORT_DIGITS) -> str:
    """'above', 'below' or 'equal to': a compared with b, at the precision shown in the report."""
    ra, rb = round(float(a), digits), round(float(b), digits)
    if ra == rb:
        return "equal to"
    return "above" if ra > rb else "below"


def compare_to_rule(summary: pd.DataFrame, target: str, method: str) -> dict:
    """For every (feature_set, metric) cell: method mean vs rule mean.
    Returns {'above'|'below'|'equal to': [(feature_set, metric, model_mean, rule_mean), ...]}."""
    out = {"above": [], "below": [], "equal to": []}
    for fs in FEATURE_SETS:
        for metric in METRIC_ORDER:
            a = mean_of(summary, target, fs, method, metric)
            b = mean_of(summary, target, RULE_FEATURE_SET, RULE_METHOD, metric)
            out[relation(a, b)].append((fs, metric, a, b))
    return out


def count_phrase(cmp: dict) -> str:
    """e.g. 'below in 12, equal in 2, above in 4 of 18 (feature set x metric) cells'."""
    n = sum(len(v) for v in cmp.values())
    parts = [f"{word.split()[0]} in {len(cmp[word])}" for word in ("below", "equal to", "above") if cmp[word]]
    return f"{', '.join(parts)} of {n} (feature set x metric) cells"


def not_beating_table(summary: pd.DataFrame, target: str) -> str:
    """Every supervised (feature_set, metric) cell whose mean is not above the rule's."""
    lines = ["| feature set | method | metric | model mean | rule mean | model is |", "|---|---|---|---|---|---|"]
    fs_order, metric_order = list(FEATURE_SETS), list(METRIC_ORDER)
    for method in ("hgb", "logreg"):
        cmp = compare_to_rule(summary, target, method)
        cells = sorted(cmp["below"] + cmp["equal to"], key=lambda c: (fs_order.index(c[0]), metric_order.index(c[1])))
        for fs, metric, a, b in cells:
            lines.append(f"| {fs} | {method} | {metric} | {a:.3f} | {b:.3f} | {relation(a, b)} the rule |")
    if len(lines) == 2:
        lines.append("| (none) | | | | | |")
    return "\n".join(lines)


def top_feature(summary: pd.DataFrame, target: str, feature_set: str):
    """(feature, mean AP drop, largest other feature's drop) from hgb permutation importance."""
    cols = FEATURE_SETS[feature_set]
    vals = {
        c: mean_of(summary, target, feature_set, "hgb", "perm_importance_ap:" + c, kind="permutation_importance")
        for c in cols
    }
    best = max(cols, key=lambda c: vals[c])  # first in FEATURE_SETS order wins ties
    others = [v for c, v in vals.items() if c != best]
    return best, vals[best], (max(others) if others else float("nan"))


def write_report(summary, data, n_splits, n_repeats, seed, path, roles=None) -> None:
    """roles: answer_key_roles() output; without it the fraud-type breakdowns are left out."""
    n = len(data)
    p, s = PRIMARY_TARGET, SECONDARY_TARGET
    pos = {t: int(data[LABEL_PREFIX + t].sum()) for t in TARGETS}
    ops = {t: rule_operating_point(data, t) for t in TARGETS}
    is_p = (data[LABEL_PREFIX + p] == 1).to_numpy()
    pay_counts = data.loc[is_p, "n_payments"].value_counts().sort_index()
    # One payment count shared by every positive and by nobody else = the label is a lookup on n_payments.
    sep_value = int(pay_counts.index[0]) if len(pay_counts) == 1 else None
    n_with_sep = int((data["n_payments"] == sep_value).sum()) if sep_value is not None else 0
    exact_fingerprint = sep_value is not None and n_with_sep == pos[p]

    flagged = rule_scores(data["n_payments"]) == 1
    other = flagged & ~is_p
    other_sec = int((other & (data[LABEL_PREFIX + s] == 1).to_numpy()).sum())
    n_other = int(other.sum())
    share = f"all {n_other}" if other_sec == n_other else f"{other_sec}"
    other_txt = f"{share} are other planted frauds in the answer key"
    if roles:
        other_txt += f": {role_breakdown(data.loc[other, 'beneficiary_id'], roles)},"
    other_pay = data.loc[other, "n_payments"]
    rule = (RULE_FEATURE_SET, RULE_METHOD)

    def m(t, fs, meth, metric):
        return mean_of(summary, t, fs, meth, metric)

    def vs(t, fs, meth, metric):
        """e.g. '0.008 vs rule 0.161 (below)'."""
        a, b = m(t, fs, meth, metric), m(t, *rule, metric)
        return f"{a:.3f} vs rule {b:.3f} ({relation(a, b)})"

    def rng(t, fs, meth, metric):
        r = summary_row(summary, t, fs, meth, metric)
        return f"{r['min']:.2f}-{r['max']:.2f}"

    # 1. How the primary label relates to the payment count.
    pay_txt = ", ".join(f"{int(k)} (x{v})" for k, v in pay_counts.items())
    if exact_fingerprint:
        f1_head = f"**n_payments = {sep_value} alone identifies the planted anomaly.**"
    else:
        f1_head = "**Payment counts of the planted anomalies.**"
    f1 = (
        f"1. {f1_head} The {pos[p]} planted anomalies have n_payments = {pay_txt}"
        + (f"; {n_with_sep} beneficiaries in total have exactly {sep_value}" if sep_value is not None else "")
        + f". The rule `n_payments > {RULE_MAX_PAYMENTS}` flags {ops[p]['flagged']} beneficiaries, catching "
        f"{ops[p]['tp']}/{ops[p]['positives']} (precision {ops[p]['precision']:.3f}). Of its other "
        f"{n_other} flags, {other_txt}"
        + (f" with n_payments {int(other_pay.min())}-{int(other_pay.max())}" if len(other_pay) else "")
        + f"; against `{s}` the rule has precision {ops[s]['precision']:.3f} and recall {ops[s]['recall']:.3f}."
    )

    # 2. All features.
    top, top_v, rest_v = top_feature(summary, p, "all")
    f2 = (
        f"2. **All features, primary target.** PR-AUC: HGB {m(p, 'all', 'hgb', 'pr_auc'):.3f}, "
        f"Isolation Forest {m(p, 'all', 'isolation_forest', 'pr_auc'):.3f}, rule {m(p, *rule, 'pr_auc'):.3f}. "
        f"HGB permutation importance: {top} mean AP drop {top_v:.3f}, largest other feature {rest_v:.3f}."
    )
    if exact_fingerprint and top == "n_payments":
        f2 += (
            f" Given finding 1, HGB is learning 'exactly {sep_value} payments', which is how the generator "
            "planted the class, not a fraud insight."
        )

    # 3. Where the supervised models do not beat the rule (counts computed cell by cell).
    cmp_hgb, cmp_lr, cmp_if = (compare_to_rule(summary, p, meth) for meth in MODEL_METHODS)
    f3 = (
        "3. **Where the supervised models do NOT beat the rule (primary target).** "
        f"Against the rule, HGB is {count_phrase(cmp_hgb)}; logistic regression is {count_phrase(cmp_lr)}; "
        f"Isolation Forest is {count_phrase(cmp_if)}. Logistic regression is monotone in each feature, so it "
        "cannot express a single payment count. "
        f"All features: PR-AUC {vs(p, 'all', 'logreg', 'pr_auc')}; at the 1% budget recall "
        f"{vs(p, 'all', 'logreg', 'recall_at_1pct')} and precision {vs(p, 'all', 'logreg', 'precision_at_1pct')}; "
        f"at the 5% budget recall {vs(p, 'all', 'logreg', 'recall_at_5pct')}. Without n_payments (`no_count`): "
        f"1% recall {vs(p, 'no_count', 'logreg', 'recall_at_1pct')}, 5% recall "
        f"{vs(p, 'no_count', 'logreg', 'recall_at_5pct')}. Every such cell is listed in "
        "'Cells where a supervised model does not beat the rule' below."
    )

    # 4. Dropping only the count.
    top4, top4_v, _ = top_feature(summary, p, "no_count")
    f4 = (
        "4. **Dropping n_payments alone.** total_amount = mean_amount x n_payments still carries the count. "
        f"HGB PR-AUC {m(p, 'no_count', 'hgb', 'pr_auc'):.3f} (all features {m(p, 'all', 'hgb', 'pr_auc'):.3f}); "
        f"top permutation-importance feature {top4} (AP drop {top4_v:.3f})."
    )

    # 5. Dropping count and total: what is left to learn.
    med = data.groupby(is_p)[["mean_amount", "income", "amount_to_income"]].median()
    hgb5 = m(p, "no_count_no_total", "hgb", "recall_at_5pct")
    lr5 = m(p, "no_count_no_total", "logreg", "recall_at_5pct")
    f5 = (
        "5. **Without count and total (`no_count_no_total`).** PR-AUC: HGB "
        f"{m(p, 'no_count_no_total', 'hgb', 'pr_auc'):.3f} (fold range {rng(p, 'no_count_no_total', 'hgb', 'pr_auc')}), "
        f"logreg {m(p, 'no_count_no_total', 'logreg', 'pr_auc'):.3f}, Isolation Forest "
        f"{m(p, 'no_count_no_total', 'isolation_forest', 'pr_auc'):.3f}, rule {m(p, *rule, 'pr_auc'):.3f}, "
        f"random ≈ {pos[p] / n:.4f}. HGB precision at 1% {vs(p, 'no_count_no_total', 'hgb', 'precision_at_1pct')}; "
        f"recall at 5% {vs(p, 'no_count_no_total', 'hgb', 'recall_at_5pct')}, and {relation(hgb5, lr5)} logreg "
        f"({lr5:.3f}). What is left to learn: median mean_amount {med.loc[True, 'mean_amount']:.0f} for planted "
        f"anomalies vs {med.loc[False, 'mean_amount']:.0f} for the rest, median income {med.loc[True, 'income']:.0f} "
        f"vs {med.loc[False, 'income']:.0f}, median amount_to_income {med.loc[True, 'amount_to_income']:.2f} vs "
        f"{med.loc[False, 'amount_to_income']:.2f}. With ~{pos[p] / n_splits:.0f} positives per fold the spread is "
        "wide."
    )

    # 6. Secondary target.
    budget_metrics = [mt for mt in METRIC_ORDER if "_at_" in mt]
    hgb_budget = {mt: relation(m(s, "all", "hgb", mt), m(s, *rule, mt)) for mt in budget_metrics}
    if all(r == "equal to" for r in hgb_budget.values()):
        budget_txt = "At both budgets HGB (all) is equal to the rule on precision and recall"
    else:
        budget_txt = "At the budgets HGB (all) vs rule: " + ", ".join(f"{mt} {r}" for mt, r in hgb_budget.items())
    rule_share = ops[s]["flagged"] / n
    saturated = [b for b in BUDGETS if ops[s]["precision"] == 1.0 and rule_share >= b]
    saturation_txt = ""
    if saturated:
        ceilings = "; ".join(
            f"{int(b * 100)}%: precision 1.000, recall ceiling {budget_ceiling(pos[s], n, b, n_splits):.3f}"
            for b in saturated
        )
        budget_names = " and ".join(f"{int(b * 100)}%" for b in saturated)
        saturation_txt = (
            f" These budget numbers are saturated, not evidence about the models: the rule flags "
            f"{rule_share:.2%} of beneficiaries with precision {ops[s]['precision']:.3f}, more than the "
            f"{budget_names} budget{'s' if len(saturated) > 1 else ''}, so any method that ranks those people "
            f"first reaches the best possible values ({ceilings}). Compare models on PR-AUC instead."
        )
    f6 = (
        f"6. **Secondary target (`{s}`).** PR-AUC: HGB (all) {vs(s, 'all', 'hgb', 'pr_auc')}, logreg (all) "
        f"{vs(s, 'all', 'logreg', 'pr_auc')}, Isolation Forest (all) {vs(s, 'all', 'isolation_forest', 'pr_auc')}. "
        f"{budget_txt} (precision at 5% {m(s, 'all', 'hgb', 'precision_at_5pct'):.3f} vs "
        f"{m(s, *rule, 'precision_at_5pct'):.3f}, recall at 5% {m(s, 'all', 'hgb', 'recall_at_5pct'):.3f} vs "
        f"{m(s, *rule, 'recall_at_5pct'):.3f}).{saturation_txt} Without count/total HGB PR-AUC is "
        f"{m(s, 'no_count_no_total', 'hgb', 'pr_auc'):.3f}. Identity fraud (duplicates, shared accounts) is not "
        "described by these per-beneficiary amount features; that is the job of the rule-based and fuzzy detectors."
    )

    lines = [
        "# Supervised baseline vs Isolation Forest vs a one-line rule",
        "",
        "Generated by `python -m src.fraud_detection.supervised "
        f"--n-splits {n_splits} --n-repeats {n_repeats} --seed {seed}` from the tracked "
        "`data/clean/*.csv` and `data/ground_truth_fraud_ids.csv`. Offline; deterministic. Every number and "
        f"every comparison word in the findings ('above', 'below', 'equal to', 'all', 'saturated') is computed "
        f"from the data at {REPORT_DIGITS} decimals; the remaining sentences are fixed methodology notes.",
        "",
        "## Setup",
        "",
        f"- Population: {n} beneficiaries with >= 1 payment (features from `isolation_forest.build_features`).",
        f"- `{p}` (primary): {pos[p]} positives ({pos[p] / n:.2%}). "
        f"`{s}` (secondary): {pos[s]} positives ({pos[s] / n:.2%}).",
        f"- {n_splits}-fold stratified CV repeated {n_repeats}x (seed {seed}) = {n_splits * n_repeats} test folds; "
        f"each `{p}` test fold holds only ~{pos[p] / n_splits:.0f} positives, so one hit moves recall by "
        f"~{n_splits / max(pos[p], 1):.2f}. Cells are `mean ± std [min-max]` over folds.",
        "- A random ranking has PR-AUC ≈ prevalence "
        f"({pos[p] / n:.4f} primary, {pos[s] / n:.4f} secondary) and ROC-AUC 0.5.",
        "- `isolation_forest` is fit on the training fold without labels (300 trees, contamination 0.05, seed 42) "
        "and scores the test fold. The rule needs no training.",
        "- Budget metrics flag the top 1% / 5% of each test fold; ties at the cut-off are split in expectation.",
        "",
        "## Plain-language findings",
        "",
        f1,
        f2,
        f3,
        f4,
        f5,
        f6,
        "",
        "Bottom line: a supervised score here is a ceiling for these features on synthetic labels, not evidence "
        "that a supervised model would work on real cases, where labels are scarce and fraud patterns move. "
        f"On the primary target HGB PR-AUC is {m(p, 'all', 'hgb', 'pr_auc'):.3f} with all features and "
        f"{m(p, 'no_count_no_total', 'hgb', 'pr_auc'):.3f} once the payment count and total are removed.",
        "",
        f"## Primary target: `{p}`",
        "",
        metric_table(summary, p),
        "",
        f"## Secondary target: `{s}`",
        "",
        metric_table(summary, s),
        "",
        "## Cells where a supervised model does not beat the rule",
        "",
        f"Means compared at {REPORT_DIGITS} decimals; both 'below' and 'equal to' cells are listed.",
        "",
        f"### `{p}`",
        "",
        not_beating_table(summary, p),
        "",
        f"### `{s}`",
        "",
        not_beating_table(summary, s),
        "",
        "## Permutation importance (HGB, test folds, drop in average precision)",
        "",
        f"### `{p}`",
        "",
        importance_table(summary, p),
        "",
        f"### `{s}`",
        "",
        importance_table(summary, s),
        "",
        "## Leakage guard",
        "",
        "Features are built by `isolation_forest.build_features` before the answer key is read; labels are "
        "attached afterwards as `y_*` columns. `assert_no_leakage` runs on every feature set inside `run_cv` and "
        "rejects answer-key columns, `y_*` labels, identifiers (`row_id`, `beneficiary_id`) and anything outside "
        "`FEATURE_COLUMNS`. Pinned by `tests/test_supervised.py`.",
        "",
        "## Caveats",
        "",
        f"- {pos[p]} positives: fold-level numbers move in steps of ~{n_splits / max(pos[p], 1):.2f} recall; "
        "trust the spread, not the third decimal.",
        f"- Isolation Forest here is fit per training fold (~{n * (n_splits - 1) / n_splits:.0f} rows), not on the "
        "whole scored population as `isolation_forest.fit_and_score` does, so its numbers are not the project's "
        "Isolation Forest run.",
        "- The CSV has no DB `row_id`; file order stands in for it (only used to pick one row per duplicated "
        "beneficiary_id). Results can differ slightly from the DB-backed Isolation Forest run.",
        "- `any_planted` labels only beneficiaries named in the answer key (victims, orphan IDs, duplicate "
        "originals and duplicates, planted anomalies). `compare_methods.build_truth` also counts hub-account "
        "holders as `involved` in shared-account fraud; "
        + (
            f"{len((roles['hub_owner'] & set(data['beneficiary_id'])) - set(data.loc[data[LABEL_PREFIX + s] == 1, 'beneficiary_id']))} "
            "of them are in this population and are labelled negative here."
            if roles and "hub_owner" in roles
            else "here they are labelled negative."
        ),
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------


def run_supervised(
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    seed: int = DEFAULT_SEED,
    out_dir=RESULTS_DIR,
    ground_truth=GROUND_TRUTH_PATH,
) -> pd.DataFrame:
    df_ben, df_disb = load_clean_inputs()
    feats = build_features(df_ben, df_disb)
    assert_no_leakage(FEATURE_COLUMNS)
    gt = pd.read_csv(ground_truth, dtype=str)
    data = attach_labels(feats, gt)
    roles = answer_key_roles(gt, load_accounts())
    for t in TARGETS:
        print(f"[supervised] {t}: {int(data[LABEL_PREFIX + t].sum())} positives of {len(data)}")

    folds, imps = [], []
    for t in TARGETS:
        f, i = run_cv(data, t, n_splits, n_repeats, seed)
        folds.append(f)
        imps.append(i)
    summary = summarise(pd.concat(folds, ignore_index=True), pd.concat(imps, ignore_index=True))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / METRICS_FILE, index=False)
    write_report(summary, data, n_splits, n_repeats, seed, out_dir / REPORT_FILE, roles=roles)
    print(f"[supervised] Wrote {out_dir / METRICS_FILE} and {out_dir / REPORT_FILE}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline supervised baseline vs Isolation Forest (Stage 7b).")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--ground-truth", type=str, default=str(GROUND_TRUTH_PATH))
    args = parser.parse_args()

    run_supervised(args.n_splits, args.n_repeats, args.seed, args.out_dir, args.ground_truth)
