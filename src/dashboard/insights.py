"""
insights.py -- turns snapshot files into plain-language explanations.

Used by BOTH app.py and report.py, so the dashboard and the saved report can
never disagree. No Streamlit and no database in here: it only reads the
snapshot folder written by publish.py.

All strings are plain text (no markdown/HTML) so they render the same in the
app and in the report. Numbers are always computed from the snapshot, never
typed in by hand.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Keep in sync with VALIDATED_SIGNALS in publish.py.
VALIDATED_SIGNALS = {
    "orphaned_disbursement",
    "shared_bank_account",
    "fuzzy_duplicate_identity",
    "multivariate_anomaly",
}


# ── LOADING ──────────────────────────────────────────────────────────────


def _read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    return pd.read_csv(path, **kwargs) if path.exists() else None


def load_snapshot_dir(d) -> dict:
    """Read every file of one snapshot folder into a dict."""
    d = Path(d)
    r = d / "results"
    report = r / "comparison_report.md"
    heatmap = r / "comparison_coverage_heatmap.png"
    return {
        "queue": pd.read_csv(d / "case_queue.csv"),
        "signals": pd.read_csv(d / "signals_long.csv"),
        "kpis": json.loads((d / "kpis.json").read_text()),
        "manifest": json.loads((d / "manifest.json").read_text()),
        "metrics": _read_csv(r / "comparison_detector_metrics.csv"),
        "coverage": _read_csv(r / "comparison_coverage_matrix.csv", index_col=0),
        "unique": _read_csv(r / "comparison_unique_catches.csv"),
        "jaccard": _read_csv(r / "comparison_overlap_jaccard.csv", index_col=0),
        "heatmap_png": str(heatmap) if heatmap.exists() else None,
        "report_md": report.read_text() if report.exists() else None,
    }


# ── SMALL HELPERS ────────────────────────────────────────────────────────

# Pakistan has no daylight saving, so a fixed offset is safe and needs no tz database.
PKT = timezone(timedelta(hours=5), "PKT")


def format_ts(iso) -> str:
    """Readable timestamp. New snapshots carry a UTC offset and are shown in Pakistan time;
    older snapshots (no offset) are shown as written."""
    try:
        dt = datetime.fromisoformat(str(iso))
    except ValueError:
        return str(iso)[:19].replace("T", " ")
    if dt.tzinfo is None:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.astimezone(PKT).strftime("%Y-%m-%d %H:%M:%S") + " PKT"


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def num(x) -> Optional[float]:
    """float or None (for NaN / blanks / text)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def pct(x, digits: int = 0) -> str:
    v = num(x)
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


def split_metrics(metrics: Optional[pd.DataFrame]):
    """(validated, heuristic) rows of comparison_detector_metrics.csv."""
    if metrics is None:
        empty = pd.DataFrame()
        return empty, empty
    if "has_ground_truth" in metrics.columns:
        mask = truthy(metrics["has_ground_truth"])
        return metrics[mask], metrics[~mask]
    return metrics, metrics.iloc[0:0]


def is_baseline(name) -> bool:
    return str(name).lower().startswith("baseline")


# ── STATIC EXPLANATIONS ──────────────────────────────────────────────────

PROJECT_SUMMARY = (
    "Governments pay welfare money to millions of people. A ghost beneficiary is a registered "
    "person who should not be paid: someone who does not really exist, who is registered more "
    "than once, or whose payments are quietly diverted into someone else's bank account. "
    "This project builds a pipeline that collects data from three separate systems (the "
    "beneficiary registry, the national ID database and the bank payment log), cleans it, "
    "stores it, and then hunts for ghost beneficiaries using three different methods. "
    "The data is synthetic because real welfare data is private. We planted known fraud in it, "
    "so every method can be graded honestly against an answer key."
)

PIPELINE_STEPS = [
    (
        "Simulated sources",
        "Three synthetic source systems are generated: the beneficiary registry, the national ID "
        "database and the bank payment log. Known fraud patterns are planted, and realistic mess "
        "(missing values, typos, mixed date formats, duplicate rows) is added on purpose.",
    ),
    (
        "Extract",
        "Each source file is loaded, checked against the columns we expect, and copied to a staging "
        "area. The beneficiary registry is the hub: if it is broken the run stops. The other two "
        "sources are spokes: if one is missing, the run continues and is marked partial.",
    ),
    (
        "Transform",
        "Data is cleaned and standardised (dates, amounts, labels). Problems are flagged in separate "
        "columns and never deleted, because deleting suspicious rows would destroy the evidence.",
    ),
    (
        "Load",
        "Clean tables are loaded into PostgreSQL. There are deliberately no foreign keys, so an "
        "orphaned payment (a payment to an ID that was never registered) can exist in the database "
        "and be detected instead of being rejected.",
    ),
    (
        "Detect",
        "Three methods search for fraud independently and write their results to one shared table, "
        "so they can be compared side by side.",
    ),
    ("Compare", "Every method is graded against the planted answer key using precision, recall and F1."),
    (
        "Orchestrate and publish",
        "Apache Airflow runs all of the above in order and checks quality gates. When a run passes, "
        "a snapshot of the results is published, and this dashboard and the written report "
        "are refreshed from it.",
    ),
]

METHODS = [
    {
        "key": "rule_based",
        "name": "Rule-based checks",
        "what": "Hand-written checks such as: several different people are paid into the same bank "
        "account, or a payment was made to an ID that was never registered.",
        "strength": "Fast and transparent. Every flag has a clear, explainable reason.",
        "weakness": "Only finds patterns someone thought of in advance. Some rules are heuristics "
        "with no answer key to grade them against.",
    },
    {
        "key": "fuzzy_linkage",
        "name": "Fuzzy record linkage",
        "what": "Compares names and addresses for close-but-not-identical matches between people "
        "with the same date of birth, to find one person registered twice under slightly "
        "different details.",
        "strength": "Finds duplicates that exact matching misses, such as typos and abbreviations.",
        "weakness": "The score threshold was tuned on the planted pairs, and the gap between true "
        "matches and non-matches was narrow. Real data would need re-tuning.",
    },
    {
        "key": "isolation_forest",
        "name": "Isolation Forest (anomaly detection)",
        "what": "A machine-learning method that looks at each beneficiary's overall payment profile "
        "and flags the ones that are statistically unusual, without being told what to look "
        "for. In layered mode it first sets aside everyone the rules and fuzzy matching "
        "already flagged, so its flags are new information.",
        "strength": "Can surface unusual combinations that no single rule covers.",
        "weakness": "Rare but legitimate behaviour also looks unusual, so many of its flags are "
        "innocent. It needs a human to review them.",
    },
]

FAMILY_EXPLANATIONS = {
    "identity": "Signs the same person may be registered under more than one identity, or that the "
    "ID does not match official records.",
    "payment_channel": "Several beneficiaries are paid into the same bank account, or one person is "
    "paid twice in a cycle.",
    "existence": "Payments made to an ID that was never registered.",
    "eligibility": "Payments made to someone whose status is Inactive or Suspended.",
    "statistical": "An unusual overall payment profile found by Isolation Forest.",
}

SIGNAL_EXPLANATIONS = {
    "shared_bank_account": "Several different beneficiaries are paid into the same bank account. (Validated against planted fraud.)",
    "orphaned_disbursement": "A payment was made to a beneficiary ID that was never registered. (Validated.)",
    "fuzzy_duplicate_identity": "Name, address and date of birth closely match another beneficiary: possibly the same person registered twice. (Validated.)",
    "multivariate_anomaly": "The payment profile is unusual across several features at once (Isolation Forest, layered mode). (Validated on planted anomalies.)",
    "cnic_mismatch": "The CNIC in the registry differs from the linked national ID record. (Heuristic, not validated; often just a data-entry typo.)",
    "duplicate_cycle_payment": "Paid twice in the same monthly cycle. (Heuristic; overlaps with shared bank account.)",
    "payment_to_inactive_beneficiary": "A payment was made to someone whose status is Inactive or Suspended. (Heuristic, not validated.)",
    "duplicate_cnic_multiple_identities": "The same CNIC appears under several beneficiary IDs. (Heuristic.)",
}

SIGNAL_SHORT = {
    "shared_bank_account": "paid into a bank account shared with other beneficiaries",
    "orphaned_disbursement": "paid under an ID that is not in the registry",
    "fuzzy_duplicate_identity": "looks like a duplicate registration of another person",
    "multivariate_anomaly": "unusual overall payment profile",
    "cnic_mismatch": "CNIC differs from the national ID record",
    "duplicate_cycle_payment": "paid twice in the same monthly cycle",
    "payment_to_inactive_beneficiary": "paid while Inactive or Suspended",
    "duplicate_cnic_multiple_identities": "same CNIC used by several IDs",
}

DQ_FLAG_EXPLANATIONS = {
    "income_flag": "monthly income unreadable, negative or implausibly high",
    "phone_missing": "phone number missing",
    "address_missing": "address missing",
    "is_duplicate_row": "exact duplicate of another row",
    "amount_flag": "payment amount unreadable, negative or above the cap",
    "payment_method_missing": "payment method missing",
    "disb_date_parse_failed": "payment date could not be read",
}

METRIC_COLUMNS = {
    "n_flagged": "How many beneficiaries the method flagged.",
    "tp_core": "How many of the planted cases it found.",
    "core_size": "How many planted cases exist for this pattern.",
    "precision": "Of everything it flagged, the share that was real planted fraud.",
    "recall": "Of all the planted fraud, the share it found.",
    "f1": "One number that balances precision and recall.",
    "precision_strict": "Precision counting only the planted culprits, not the people around them.",
    "any_fraud_precision": "Share of flags that belong to ANY planted fraud pattern, not just the one it targets.",
    "overall_core_recall": "Share of all planted culprits across every pattern that this method found.",
}

# ── DISPLAY NAMES ────────────────────────────────────────────────────────
# The one place that turns internal names into what an investigator reads.
# Applied only when something is shown: the snapshot CSVs, the downloads and
# every computation keep the raw names. Unknown names are shown unchanged.

COLUMN_LABELS = {
    # comparison_detector_metrics.csv
    "detector": "Detector",
    "family": "Evidence family",
    "level": "Graded as",
    "has_ground_truth": "Has answer key",
    "target": "Planted pattern targeted",
    "n_flagged": "Beneficiaries flagged",
    "tp_core": "Planted cases found",
    "core_size": "Planted cases",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 score",
    "precision_strict": "Strict precision",
    "any_fraud_precision": "Flags that are any planted fraud",
    "overall_core_recall": "Share of all planted fraud found",
    # case_queue.csv
    "rank": "Rank",
    "beneficiary_id": "Beneficiary ID",
    "n_validated_families": "Validated families",
    "n_families": "Total families",
    "families": "Evidence families",
    "methods": "Methods",
    "signal_types": "Signals",
    "evidence_score": "Evidence score",
    "max_score": "Highest score",
    "has_validated_signal": "Has a validated signal",
    "total_flags": "Total flags",
    "in_registry": "In registry",
    "why_flagged": "Why flagged",
    # signals_long.csv and kpis.json
    "method": "Method",
    "signal_type": "Signal",
    "score": "Score",
    "n_flags": "Flags",
    "n_beneficiaries": "Beneficiaries flagged",
    # comparison coverage / unique catches / overlap
    "planted_pattern": "Planted pattern",
    "core_caught": "Planted cases caught",
    "caught_by_this_method_only": "Caught by this method only",
    "core_total": "Planted cases in total",
    # report tables built from kpis.json
    "table": "Table",
    "rows": "Rows",
    "flag": "Data-quality check",
    "meaning": "Meaning",
    "share_of_rows": "Share of rows",
    "validated": "Validated",
}

# Header for a named row index. The coverage and overlap matrices are read with
# index_col=0, so their first CSV header becomes the index name. The overlap
# file's first header is "jaccard", but its rows are methods.
INDEX_LABELS = {
    "jaccard": "Method",
    "planted_pattern": "Planted pattern",
}

# In the metrics table "family" is the kind of method, not the kind of evidence.
METRIC_COLUMN_LABELS = {**COLUMN_LABELS, "family": "Method type"}

METHOD_LABELS = {
    # method keys in signals_long.csv / case_queue.csv
    "rule_based": "Rule-based checks",
    "fuzzy_linkage": "Fuzzy record linkage",
    "isolation_forest": "Isolation Forest",
    # method types in comparison_detector_metrics.csv
    "fuzzy": "Fuzzy record linkage",
    "baseline": "Baseline",
    "ensemble": "Combined methods",
    # single detectors (graded one signal at a time)
    "rule:orphaned_disbursement": "Rule: orphaned payment",
    "rule:shared_bank_account": "Rule: shared bank account",
    "rule:duplicate_cnic_multiple_identities": "Rule: same CNIC, several IDs",
    "rule:cnic_mismatch": "Rule: CNIC mismatch",
    "rule:duplicate_cycle_payment": "Rule: paid twice in one cycle",
    "rule:payment_to_inactive_beneficiary": "Rule: payment to inactive beneficiary",
    "fuzzy:duplicate_identity": "Fuzzy matching: duplicate identity",
    "iforest:standard": "Isolation Forest (standard)",
    "iforest:layered": "Isolation Forest (layered)",
    "baseline:payment_count_gt_6": "Baseline: more than 6 payments",
    # whole methods (comparison tables). The ", whole method" suffix keeps each
    # apart from the single detector of the same name in the same table.
    "METHOD rule_based (validated 2)": "Rule-based checks (2 validated rules), whole method",
    "METHOD rule_based (all 6)": "Rule-based checks (all 6 rules), whole method",
    "METHOD fuzzy linkage": "Fuzzy record linkage, whole method",
    "METHOD isolation_forest (standard)": "Isolation Forest (standard), whole method",
    "METHOD isolation_forest (layered)": "Isolation Forest (layered), whole method",
    "METHOD ENSEMBLE (rules2 + fuzzy + IF)": "Combined (2 validated rules + fuzzy + Isolation Forest), whole method",
}

SIGNAL_LABELS = {
    "shared_bank_account": "Shared bank account",
    "orphaned_disbursement": "Orphaned payment (ID not registered)",
    "fuzzy_duplicate_identity": "Possible duplicate identity",
    "multivariate_anomaly": "Unusual payment profile",
    "cnic_mismatch": "CNIC mismatch",
    "duplicate_cycle_payment": "Paid twice in one cycle",
    "payment_to_inactive_beneficiary": "Payment to inactive beneficiary",
    "duplicate_cnic_multiple_identities": "Same CNIC, several IDs",
}

PATTERN_LABELS = {
    "orphan_disbursement": "Orphaned payments",
    "shared_account": "Shared bank accounts",
    "duplicate_identity": "Duplicate identities",
    "multivariate_anomaly": "Unusual payment profiles",
    "-": "None (no answer key)",
}

FAMILY_LABELS = {
    "identity": "Identity",
    "payment_channel": "Payment channel",
    "existence": "Existence",
    "eligibility": "Eligibility",
    "statistical": "Statistical",
}

LEVEL_LABELS = {"signal": "Single signal", "method": "Whole method"}

TABLE_LABELS = {
    "beneficiaries": "Beneficiaries",
    "national_id_records": "National ID records",
    "disbursements": "Payments",
    "fraud_signals": "Fraud signals",
    "record_linkage_matches": "Record-linkage matches",
}

DQ_FLAG_LABELS = {
    "income_flag": "Income problem",
    "phone_missing": "Phone missing",
    "address_missing": "Address missing",
    "is_duplicate_row": "Duplicate row",
    "amount_flag": "Amount problem",
    "payment_method_missing": "Payment method missing",
    "disb_date_parse_failed": "Payment date unreadable",
}

# Which value mapping applies to which column. "multivariate_anomaly" is both a
# signal and a planted pattern, so each column looks in its own dictionary.
VALUE_LABELS = {
    "detector": METHOD_LABELS,
    "method": METHOD_LABELS,
    "methods": METHOD_LABELS,
    "jaccard": METHOD_LABELS,  # row index of the overlap matrix (see INDEX_LABELS)
    "signal_type": SIGNAL_LABELS,
    "signal_types": SIGNAL_LABELS,
    "family": {**FAMILY_LABELS, **METHOD_LABELS},
    "families": FAMILY_LABELS,
    "target": PATTERN_LABELS,
    "planted_pattern": PATTERN_LABELS,
    "level": LEVEL_LABELS,
    "table": TABLE_LABELS,
    "flag": DQ_FLAG_LABELS,
}

# Columns holding a 0-1 share, shown as a percentage. Jaccard overlap is a
# similarity score, not a share, so it is left as a decimal.
PERCENT_COLUMNS = {
    "precision",
    "recall",
    "f1",
    "precision_strict",
    "any_fraud_precision",
    "overall_core_recall",
    "share_of_rows",
}

# Columns of whole-number counts (shown without decimals even when stored as floats).
COUNT_COLUMNS = {
    "n_flagged",
    "tp_core",
    "core_size",
    "rank",
    "n_validated_families",
    "n_families",
    "total_flags",
    "n_flags",
    "n_beneficiaries",
    "core_caught",
    "caught_by_this_method_only",
    "core_total",
}


# Headers for the "Weight each signal gets in the ranking" table (kpis.json ranking_weights).
WEIGHT_COLUMN_LABELS = {
    "signal_type": "Signal",
    "detector": "Detector",
    "source": "Weight source",
    "column": "Precision reading",
    "n_flagged": "Flagged",
    "precision": "Measured precision",
    "weight": "Weight in ranking",
}


def label(name, mapping: Optional[dict] = None) -> str:
    """Display name for one internal name. Without a mapping, every known name is tried."""
    key = str(name).strip()
    if mapping is not None:
        return mapping.get(key, key)
    for m in (COLUMN_LABELS, METHOD_LABELS, SIGNAL_LABELS, PATTERN_LABELS, FAMILY_LABELS, TABLE_LABELS):
        if key in m:
            return m[key]
    return DQ_FLAG_LABELS.get(key, key)


def label_list(names, mapping: dict) -> str:
    """Relabel a comma-separated cell such as "fuzzy_linkage, rule_based"."""
    if names is None or (isinstance(names, float) and names != names):
        return ""
    return ", ".join(label(n, mapping) for n in str(names).split(",") if n.strip())


def display_frame(df: pd.DataFrame, columns: Optional[dict] = None) -> pd.DataFrame:
    """Copy of df with readable values and column headers. Numbers are left untouched.

    columns: the header mapping to use (COLUMN_LABELS by default). A named index
    (as in the coverage and overlap matrices) gets its values relabelled and its
    name taken from INDEX_LABELS."""
    columns = columns or COLUMN_LABELS
    out = df.copy()
    for col in out.columns:
        mapping = VALUE_LABELS.get(col)
        if mapping is None or pd.api.types.is_numeric_dtype(out[col]):
            continue
        if col in ("methods", "signal_types", "families"):
            out[col] = out[col].map(lambda s, m=mapping: label_list(s, m))
        else:
            out[col] = out[col].map(lambda s, m=mapping: s if pd.isna(s) else label(s, m))
    if out.index.name in VALUE_LABELS:
        out.index = out.index.map(lambda s: label(s, VALUE_LABELS[out.index.name]))
    if out.index.name is not None:
        out.index.name = INDEX_LABELS.get(out.index.name, columns.get(out.index.name, out.index.name))
    out.columns = [columns.get(c, label(c, METHOD_LABELS)) for c in out.columns]
    return out


BAG_ANALOGY = (
    "Think of airport security checking bags. Precision asks: of all the bags the officers pulled "
    "aside, how many really contained something? Recall asks: of all the bags that really contained "
    "something, how many did the officers catch? An officer who only stops bags they are certain "
    "about has high precision but lets many dangerous bags through (low recall). An officer who "
    "stops almost every bag catches nearly everything dangerous (high recall) but wastes huge amounts "
    "of time on innocent travellers (low precision). Fraud detection has the same trade-off."
)

GLOSSARY = {
    "Beneficiary": "A person enrolled in a welfare programme who receives payments.",
    "Ghost beneficiary": "A registered person who should not be paid: fake, duplicated, or with payments diverted elsewhere.",
    "Signal (flag)": "One warning raised by one method about one beneficiary or payment. A flag is a lead for a human, not a verdict.",
    "Evidence family": "A group of related signals that count as one piece of evidence, because they tend to fire for the same reason. The five families are identity, payment channel, existence, eligibility and statistical.",
    "Validated signal": "A signal whose method was graded against planted fraud (orphaned payment, shared bank account, fuzzy duplicate, layered Isolation Forest). Other signals are heuristics.",
    "Ground truth (answer key)": "The list of fraud cases we planted ourselves in the synthetic data, used to grade each method honestly.",
    "Precision": "Of everything a method flagged, the share that was real fraud.",
    "Recall": "Of all the real fraud, the share a method found.",
    "F1": "A single score that balances precision and recall.",
    "Orphaned disbursement": "A payment made to a beneficiary ID that does not exist in the registry.",
    "Shared bank account": "One bank account receiving payments for several different beneficiary IDs.",
    "Fuzzy matching": "Comparing text that is similar but not identical, such as names with a typo.",
    "Isolation Forest": "A machine-learning method that isolates unusual records without being told what fraud looks like.",
    "CNIC": "Computerised National Identity Card number, the main identity number in the registry.",
    "Snapshot": "A frozen copy of the results of one successful pipeline run. The dashboard and report read only snapshots, so a run in progress can never blank them.",
    "Airflow DAG": "The ordered schedule of pipeline tasks that Apache Airflow runs automatically.",
}


# ── DYNAMIC (NUMBER-DRIVEN) TEXT ─────────────────────────────────────────


def describe_detector(row: pd.Series) -> str:
    """One plain-language sentence group for one row of the metrics table."""
    raw = str(row.get("detector"))
    name = label(raw, METHOD_LABELS)
    n_flagged = num(row.get("n_flagged"))
    tp = num(row.get("tp_core"))
    core = num(row.get("core_size"))
    precision = num(row.get("precision"))
    recall = num(row.get("recall"))

    flagged_txt = f"flagged {int(n_flagged):,} beneficiaries" if n_flagged is not None else "was evaluated"
    if is_baseline(raw):
        found = f" and found {int(tp)} of {int(core)} planted cases" if tp is not None and core else ""
        return (
            f"{name} is a one-line reference rule, not a proposed detector. It {flagged_txt}{found}. "
            "It is included only to show how a very simple rule compares."
        )

    parts = [f"{name} {flagged_txt}."]
    if tp is not None and core:
        parts.append(
            f"Of the {int(core)} planted cases it was graded on, it found {int(tp)}"
            + (f" (recall {pct(recall)})." if recall is not None else ".")
        )
    if precision is not None:
        extra = f", roughly 1 in {1 / precision:.1f} flags" if 0 < precision < 0.5 else ""
        parts.append(f"{pct(precision)} of its flags were planted fraud{extra}.")
    return " ".join(parts)


def executive_summary(data: dict) -> list:
    q, m = data["queue"], data["manifest"]
    ts = format_ts(m["published_at"])
    lines = [
        f"This summary describes pipeline snapshot {m['snapshot_id']}, published {ts}.",
        f"The system flagged {len(q):,} beneficiaries in total.",
        f"{int((q['n_families'] >= 2).sum()):,} of them were flagged by two or more independent kinds "
        f"of evidence, and {int((q['n_validated_families'] >= 2).sum()):,} by two or more kinds of "
        "evidence that were validated against planted fraud. Those are the strongest leads.",
    ]
    n_unreg = int((~q["in_registry"]).sum())
    if n_unreg:
        lines.append(
            f"Flagged IDs that do not exist in the registry at all: {n_unreg:,}. Payments were made "
            "against IDs that were never registered."
        )

    validated, _ = split_metrics(data.get("metrics"))
    real = [r for _, r in validated.iterrows() if not is_baseline(r.get("detector"))]
    if real:
        perfect = [
            str(r["detector"])
            for r in real
            if (num(r.get("recall")) or 0) >= 0.99 and (num(r.get("precision")) or 0) >= 0.99
        ]
        others = [r for r in real if str(r["detector"]) not in perfect]
        lines.append(f"{len(real)} detectors were graded against the planted answer key.")
        if perfect:
            lines.append(
                "Found every planted case with no false alarms: "
                + ", ".join(label(p, METHOD_LABELS) for p in perfect)
                + "."
            )
        for r in others:
            lines.append(describe_detector(r))
    lines.append(
        "Perfect scores only cover the fraud patterns planted in this synthetic data. "
        "They are not proof the methods would work as well on real data."
    )
    return lines


def queue_takeaway(data: dict) -> str:
    """One sentence on the strongest cases, computed from the queue's actual ranks
    (so it stays true whatever the ranking rule puts first)."""
    q = data["queue"]
    ranks = q["rank"] if "rank" in q.columns else pd.Series(range(1, len(q) + 1), index=q.index)
    multi = ranks[q["n_validated_families"] >= 2]
    if "evidence_score" in q.columns:
        rule = (
            "The queue is ordered by evidence score (how often each signal is right, with independent kinds "
            "of evidence combined), so one very reliable signal can rank above two weaker ones."
        )
    else:
        rule = (
            "This older snapshot is ordered by how many validated kinds of evidence flag a person, "
            "then how many kinds in total."
        )
    if multi.empty:
        return f"No beneficiary is flagged by two or more validated kinds of evidence. {rule}"
    lo, hi = int(multi.min()), int(multi.max())
    if len(multi) == 1:
        head = f"1 beneficiary is flagged by two or more validated kinds of evidence; it is at rank {lo:,}"
    else:
        head = (
            f"{len(multi):,} beneficiaries are flagged by two or more validated kinds of evidence; "
            f"the highest of them is at rank {lo:,} and the lowest at rank {hi:,}"
        )
    return f"{head} (of {len(q):,}). {rule}"


def why_flagged(signal_types: str) -> str:
    return "; ".join(SIGNAL_SHORT.get(s.strip(), s.strip()) for s in str(signal_types).split(","))


def limitations(data: dict) -> list:
    out = [
        "Perfect or near-perfect scores only cover the fraud patterns planted in this synthetic data. "
        "They are not evidence that the methods would perform the same way on real welfare data.",
    ]
    validated, _ = split_metrics(data.get("metrics"))
    for _, r in validated.iterrows():
        name = str(r.get("detector")).lower()
        p = num(r.get("precision"))
        if ("isolation" in name or "iforest" in name) and "layer" in name and p:
            out.append(
                f"The layered Isolation Forest is right about 1 in {1 / p:.1f} of its flags "
                f"(precision {pct(p)}). Its flags are leads to review, not conclusions."
            )
    out += [
        "The anomalies were planted with a clean separator (a beneficiary with seven payments where "
        "normal beneficiaries have at most six). That is why a one-line rule can match Isolation Forest. "
        "Real data would not offer such a clean separator.",
        "The fuzzy-matching threshold was tuned on the same 40 planted pairs it is graded on, and the gap "
        "between true pairs and the best non-match was only about 0.016 when it was tuned. Real-world name "
        "variation would need re-tuning.",
        "Four rules (CNIC mismatch, duplicate cycle payment, payment to inactive beneficiary and duplicate CNIC) "
        "have no answer key, so they are heuristics. For example, many CNIC mismatches here come from typos the "
        "data generator injected on purpose. Treat their flags as leads, not proof.",
        "Every flag still needs a human investigator. The system ranks and explains; it does not decide.",
    ]
    return out
