"""
app.py -- Stage 10b: Streamlit dashboard for the welfare fraud pipeline.

Reads ONLY the snapshot written by publish.py (data/dashboard/latest.json
points at the newest snapshot folder). It never touches Postgres, so a DAG
run that truncates the tables cannot blank the dashboard.

All plain-language text and number-driven sentences live in insights.py, and
the saved report is built by report.py from the same snapshot files, so the
dashboard and the report always tell the same story.

Run from the project root (or anywhere):
    streamlit run src/dashboard/app.py

Needs:  pip install streamlit plotly pandas
"""

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]  # src/dashboard/app.py -> project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard import insights as ins  # noqa: E402
from src.dashboard.evaluation import (  # noqa: E402
    MIN_ANSWER_KEY_COVERAGE,
    answer_key_coverage,
    load_planted_ids,
    precision_at_k,
)
from src.dashboard.report import build_report  # noqa: E402

SNAPSHOT_ROOT = ROOT / "data" / "dashboard"
POINTER = SNAPSHOT_ROOT / "latest.json"
REFRESH_EVERY = "30s"


# ── SNAPSHOT LOADING ─────────────────────────────────────────────────────


def read_pointer():
    """Return the newest snapshot_id, or None if nothing is published yet."""
    if not POINTER.exists():
        return None
    try:
        return json.loads(POINTER.read_text())["snapshot_id"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


@st.cache_data(show_spinner=False, max_entries=3)
def load_snapshot(snapshot_id: str) -> dict:
    """Cached per snapshot_id: a new snapshot means a cache miss and a fresh load."""
    return ins.load_snapshot_dir(SNAPSHOT_ROOT / snapshot_id)


def explain(title: str, lines, expanded: bool = True):
    """A 'what am I looking at?' box. Open by default so it can be read out while presenting."""
    with st.expander(title, expanded=expanded):
        for line in lines:
            st.write(line)


# ── DISPLAY HELPERS ──────────────────────────────────────────────────────
# Names come from the shared mapping in insights.py; these only format. The
# data passed in is never modified, so downloads keep the raw snapshot schema.


def show_table(df: pd.DataFrame, columns: Optional[dict] = None, column_config: Optional[dict] = None, **kwargs):
    """st.dataframe with readable headers and values, shares as percentages and counts as whole numbers."""
    columns = columns or ins.COLUMN_LABELS
    config = {}
    for c in df.columns:
        name = columns.get(c, c)
        if c in ins.PERCENT_COLUMNS:
            config[name] = st.column_config.NumberColumn(name, format="percent", help=ins.METRIC_COLUMNS.get(c))
        elif c in ins.COUNT_COLUMNS:
            config[name] = st.column_config.NumberColumn(name, format="%d", help=ins.METRIC_COLUMNS.get(c))
    config.update(column_config or {})
    st.dataframe(ins.display_frame(df, columns), column_config=config, **kwargs)


def whole_number_axis(fig, axis: str, max_value):
    """Count axes: no 0.5 ticks. Small ranges get a tick on every integer."""
    update = fig.update_xaxes if axis == "x" else fig.update_yaxes
    update(tickformat=",d", rangemode="tozero")
    if max_value is not None and max_value <= 10:
        update(dtick=1)


# ── PAGE 0: START HERE ───────────────────────────────────────────────────


def page_start(data: dict, snapshot_id: str):
    st.title("Welfare fraud detection")
    st.write(ins.PROJECT_SUMMARY)

    st.subheader("The 30-second story")
    for line in ins.executive_summary(data):
        st.markdown(f"- {line}")

    st.subheader("How the system works")
    for i, (title, text) in enumerate(ins.PIPELINE_STEPS, 1):
        st.markdown(f"**{i}. {title}.** {text}")

    st.subheader("Three ways of looking for fraud")
    cols = st.columns(3)
    for col, m in zip(cols, ins.METHODS):
        with col:
            st.markdown(f"#### {m['name']}")
            st.write(m["what"])
            st.markdown(f"**Good at:** {m['strength']}")
            st.markdown(f"**Weak at:** {m['weakness']}")

    st.subheader("How to use this dashboard")
    st.markdown(
        "- **Overview**: how healthy the run was and how much each method flagged.\n"
        "- **Method comparison**: how well each method did against the planted answer key.\n"
        "- **Investigation queue**: the ranked list of people to investigate, with reasons.\n"
        "- **Fraud explorer**: everything each method flagged, filterable.\n"
        "- **Report**: an automatically written report you can download and hand in.\n"
        "- **Glossary**: plain-language meaning of every term."
    )
    with st.expander("Honest limitations"):
        for line in ins.limitations(data):
            st.markdown(f"- {line}")


# ── PAGE 1: OVERVIEW ─────────────────────────────────────────────────────


def page_overview(data: dict, snapshot_id: str):
    m, k = data["manifest"], data["kpis"]
    st.title("Overview")
    explain(
        "What am I looking at?",
        [
            "This page is the health check of one pipeline run: how much data went in, how messy it was, "
            "and how many beneficiaries each method flagged.",
            "The numbers come from a snapshot, a frozen copy of the results of the last successful run.",
        ],
    )
    st.info(ins.queue_takeaway(data))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Snapshot", m["snapshot_id"])
    c2.metric("Published", ins.format_ts(m["published_at"]))
    c3.metric("Beneficiaries in queue", f"{k['queue_size']:,}")
    c4.metric("IDs not in registry", f"{k['not_in_registry']:,}")
    st.caption(
        "Snapshot ids are UTC timestamps (year, month, day, hour, minute, second). "
        "The Published time is shown in Pakistan time."
    )
    with st.expander("Run details (which pipeline run produced this)"):
        st.json({"dag_run_id": m.get("dag_run_id"), "run_ids": m.get("run_ids")})

    st.subheader("Rows in the database when this snapshot was built")
    st.caption("Counts include exact duplicate rows that were created on purpose as data-quality noise.")
    cols = st.columns(len(k["row_counts"]))
    for col, (table, n) in zip(cols, k["row_counts"].items()):
        col.metric(ins.label(table, ins.TABLE_LABELS), f"{n:,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Data-quality flag rates")
        st.caption(
            "The source data is deliberately messy, like real government data. Problems are flagged, never deleted."
        )
        rows = [
            {"table": ins.label(t, ins.TABLE_LABELS), "flag": ins.label(f, ins.DQ_FLAG_LABELS), "rate": v}
            for t, flags in k["dq_flag_rates"].items()
            for f, v in flags.items()
        ]
        fig = px.bar(
            pd.DataFrame(rows),
            x="rate",
            y="flag",
            color="table",
            orientation="h",
            barmode="group",
            text_auto=".1%",
            labels={"rate": "Share of rows flagged", "flag": "Data-quality check", "table": "Table"},
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig)
        with st.expander("What each flag means"):
            for flag, meaning in ins.DQ_FLAG_EXPLANATIONS.items():
                st.markdown(f"- **{ins.label(flag, ins.DQ_FLAG_LABELS)}**: {meaning}")
    with right:
        st.subheader("Beneficiaries flagged, by signal")
        st.caption(
            "Each bar is the number of distinct beneficiaries one signal flagged. The same person can appear in several bars."
        )
        sig = ins.display_frame(pd.DataFrame(k["signals_by_type"]))
        fig = px.bar(
            sig,
            x=ins.label("n_beneficiaries"),
            y=ins.label("signal_type"),
            color=ins.label("method"),
            orientation="h",
        )
        whole_number_axis(fig, "x", sig[ins.label("n_beneficiaries")].max())
        st.plotly_chart(fig)

    st.subheader("How many kinds of evidence flag each beneficiary?")
    st.caption("One kind of evidence is a weak lead. Two or more independent kinds is much stronger.")
    fam = pd.DataFrame(
        {
            "Kinds of evidence": [int(n) for n in k["queue_by_families"].keys()],
            "Beneficiaries": list(k["queue_by_families"].values()),
        }
    )
    fig = px.bar(fam, x="Kinds of evidence", y="Beneficiaries", text_auto=",d")
    whole_number_axis(fig, "x", fam["Kinds of evidence"].max())
    whole_number_axis(fig, "y", fam["Beneficiaries"].max())
    st.plotly_chart(fig)


# ── PAGE 2: METHOD COMPARISON ────────────────────────────────────────────


def _pick(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]]


def page_comparison(data: dict, snapshot_id: str):
    st.title("Method comparison")
    explain(
        "What am I looking at?",
        [
            "We planted known fraud in the data, so we can grade every method honestly. This page shows how "
            "each method did. Precision and recall are the two grades.",
            ins.BAG_ANALOGY,
        ],
    )

    metrics = data["metrics"]
    if metrics is None:
        st.warning("comparison_detector_metrics.csv was not found in this snapshot.")
        return
    validated, heuristic = ins.split_metrics(metrics)

    st.subheader("Graded against the planted answer key")
    show_table(
        _pick(
            validated,
            [
                "detector",
                "family",
                "level",
                "target",
                "n_flagged",
                "tp_core",
                "core_size",
                "precision",
                "recall",
                "f1",
                "precision_strict",
                "any_fraud_precision",
                "overall_core_recall",
            ],
        ),
        columns=ins.METRIC_COLUMN_LABELS,
        hide_index=True,
    )
    with st.expander("What each column means"):
        for col, meaning in ins.METRIC_COLUMNS.items():
            st.markdown(f"- **{ins.label(col, ins.METRIC_COLUMN_LABELS)}**: {meaning}")

    if {"precision", "recall", "f1"} <= set(validated.columns) and not validated.empty:
        plot = validated.copy()
        label = plot["detector"].map(lambda d: ins.label(d, ins.METHOD_LABELS))
        if "target" in plot.columns:
            label = label + " (" + plot["target"].map(lambda t: ins.label(t, ins.PATTERN_LABELS)) + ")"
        plot["label"] = label
        long = plot.melt(
            id_vars="label", value_vars=["precision", "recall", "f1"], var_name="metric", value_name="value"
        )
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long["metric"] = long["metric"].map(lambda c: ins.label(c, ins.COLUMN_LABELS))
        fig = px.bar(
            long,
            x="label",
            y="value",
            color="metric",
            barmode="group",
            range_y=[0, 1],
            labels={"label": "Detector (planted pattern it targets)", "value": "Score", "metric": "Grade"},
            hover_data={"value": ":.1%"},
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig)

    if not validated.empty:
        st.subheader("In plain words")
        for _, r in validated.iterrows():
            st.markdown(f"- {ins.describe_detector(r)}")

    st.subheader("Heuristic rules (no answer key, so no precision or recall)")
    st.caption("These are useful leads, but nothing planted lets us grade them.")
    show_table(
        _pick(heuristic, ["detector", "family", "level", "n_flagged", "any_fraud_precision"]),
        columns=ins.METRIC_COLUMN_LABELS,
        hide_index=True,
    )

    st.subheader("Which method catches which planted pattern (recall)")
    st.caption(
        "Each cell is the share of that planted pattern the method found. Dark means it finds it, light means it misses it."
    )
    coverage = data["coverage"]
    shown = False
    if coverage is not None:
        try:
            matrix = ins.display_frame(coverage.apply(pd.to_numeric, errors="coerce"))
            fig = px.imshow(
                matrix,
                text_auto=".0%",
                aspect="auto",
                zmin=0,
                zmax=1,
                color_continuous_scale="Blues",
                labels={"x": "Method", "y": "Planted pattern", "color": "Recall"},
            )
            fig.update_coloraxes(colorbar_tickformat=".0%")
            st.plotly_chart(fig)
            shown = True
        except Exception:
            shown = False
    if not shown and data["heatmap_png"]:
        st.image(data["heatmap_png"])

    with st.expander("Unique catches and overlap between methods"):
        if data["unique"] is not None:
            st.caption("Entities caught by only one method")
            show_table(data["unique"], hide_index=True)
        if data["jaccard"] is not None:
            st.caption("Overlap (Jaccard similarity) between methods: 0 means no shared catches, 1 means identical.")
            jaccard = ins.display_frame(data["jaccard"])
            st.dataframe(
                jaccard, column_config={c: st.column_config.NumberColumn(format="%.2f") for c in jaccard.columns}
            )

    st.subheader("How to read these results honestly")
    for line in ins.limitations(data):
        st.markdown(f"- {line}")

    if data["report_md"]:
        with st.expander("Technical report from the pipeline (internal names)"):
            st.caption(
                "This is the raw report the pipeline writes, shown word for word. It uses the internal names "
                "(for example rule:..., iforest:..., n_flagged, tp_core); the tables above show the same results "
                "with plain labels, and the Glossary page explains the terms."
            )
            st.markdown(data["report_md"])


# ── PAGE 3: INVESTIGATION QUEUE ──────────────────────────────────────────


def page_queue(data: dict, snapshot_id: str):
    q = data["queue"]
    st.title("Investigation queue")
    explain(
        "What am I looking at?",
        [
            "This is the list of people an investigator should look at, most suspicious first. Every flagged "
            "beneficiary appears once.",
            "Signals that tend to fire for the same reason are grouped into five kinds of evidence (families), "
            "and scores are never added up, so correlated signals cannot inflate a case.",
            "Ranking: each signal is weighted by how often it is right. Signals graded against the planted "
            "answer key use their measured precision (counting a hit the same way the answer key names fraud); "
            "heuristic signals with no answer key get a low, fixed prior instead. Within a kind of evidence only "
            "the strongest signal counts, and the kinds are combined so that independent evidence adds up but can "
            "never exceed 100% (a noisy-OR). The result is "
            "the Evidence score column. Ties are broken by how many validated kinds of evidence flag the person, "
            "then how many kinds in total, then their highest score.",
        ],
    )
    st.info(ins.queue_takeaway(data))

    st.subheader("How good is the top of the queue?")
    st.caption("Precision@k, measured against the planted answer key")
    planted = load_planted_ids(ROOT / "data" / "ground_truth_fraud_ids.csv")
    if planted is None:
        st.caption("The planted answer key (data/ground_truth_fraud_ids.csv) was not found, so this is not shown.")
    else:
        coverage = answer_key_coverage(q, planted)
        if coverage < MIN_ANSWER_KEY_COVERAGE:
            st.warning(
                f"Only {ins.pct(coverage, 1)} of the planted ids appear in this queue, so the answer key in "
                "data/ground_truth_fraud_ids.csv probably comes from a different generator run than this "
                "snapshot. The numbers below are not meaningful for it."
            )
        pk = precision_at_k(q, planted)
        st.dataframe(
            pd.DataFrame(
                {
                    "top k of the queue": pk["k"],
                    "planted fraud in top k": pk["n_planted"],
                    "precision@k": pk["precision"].map(lambda p: ins.pct(p, 1)),
                }
            ),
            hide_index=True,
        )
        st.caption(
            "Of the first k people in the queue, the share that are planted fraud. The answer key is used here "
            "for grading only, never for ranking. But the signal weights are precisions measured on the same "
            "answer key (and the cnic_mismatch prior was chosen after looking at these results), so these "
            f"numbers are in-sample and optimistic. {ins.pct(coverage, 1)} of the planted ids appear somewhere "
            "in this queue."
        )
    weights = data["kpis"].get("ranking_weights")
    if weights:
        with st.expander("Weight each signal gets in the ranking"):
            st.caption(
                "Weight source 'measured': precision graded against the planted answer key ('Precision reading' "
                "says which Stage 8 reading), smoothed so a small perfect sample is not treated as certainty. "
                "Weight source 'prior': a fixed, deliberately low guess for a heuristic with no answer key."
            )
            show_table(pd.DataFrame(weights), columns=ins.WEIGHT_COLUMN_LABELS, hide_index=True)
    with st.expander("The five kinds of evidence"):
        for fam, meaning in ins.FAMILY_EXPLANATIONS.items():
            st.markdown(f"- **{ins.label(fam, ins.FAMILY_LABELS)}**: {meaning}")

    all_families = sorted(
        {f.strip() for s in q["families"] for f in str(s).split(",")}, key=lambda f: ins.label(f, ins.FAMILY_LABELS)
    )
    all_methods = sorted(
        {m.strip() for s in q["methods"] for m in str(s).split(",")}, key=lambda m: ins.label(m, ins.METHOD_LABELS)
    )

    c1, c2, c3, c4 = st.columns(4)
    min_val = c1.slider("Min validated families", 0, max(int(q["n_validated_families"].max()), 1), 0)
    min_fam = c2.slider("Min total families", 1, max(int(q["n_families"].max()), 2), 1)
    need_fams = c3.multiselect(
        "Must include family", all_families, format_func=lambda f: ins.label(f, ins.FAMILY_LABELS)
    )
    need_methods = c4.multiselect(
        "Must include method", all_methods, format_func=lambda m: ins.label(m, ins.METHOD_LABELS)
    )
    d1, d2 = st.columns([1, 2])
    only_unreg = d1.checkbox("Only IDs not in registry")
    search = d2.text_input("Search beneficiary ID")

    view = q[(q["n_validated_families"] >= min_val) & (q["n_families"] >= min_fam)]
    for f in need_fams:
        view = view[view["families"].str.contains(f, regex=False)]
    for m in need_methods:
        view = view[view["methods"].str.contains(m, regex=False)]
    if only_unreg:
        view = view[~view["in_registry"]]
    if search:
        view = view[view["beneficiary_id"].str.contains(search.strip(), case=False, regex=False)]
    view = view.assign(why_flagged=view["signal_types"].map(ins.why_flagged))

    st.write(f"Showing **{len(view):,}** of {len(q):,} beneficiaries")
    score_col = ins.label("max_score")
    evidence_col = ins.label("evidence_score")
    show_table(
        view,
        hide_index=True,
        column_config={
            evidence_col: st.column_config.ProgressColumn(evidence_col, min_value=0.0, max_value=1.0, format="%.4f"),
            score_col: st.column_config.ProgressColumn(score_col, min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )
    st.download_button("Download filtered queue (CSV)", view.to_csv(index=False), "case_queue_filtered.csv", "text/csv")

    st.subheader("Case detail")
    options = view["beneficiary_id"].head(500).tolist()
    if not options:
        st.caption("No beneficiaries match the filters.")
        return
    chosen = st.selectbox("Beneficiary (top 500 of the current filter)", options)
    row = view[view["beneficiary_id"] == chosen].iloc[0]
    if not row["in_registry"]:
        st.warning(
            "This ID is not in the beneficiary registry: a payment was made against an ID that was never registered."
        )

    detail = data["signals"][data["signals"]["beneficiary_id"] == chosen]
    show_table(detail[["method", "signal_type", "family", "score", "n_flags"]], hide_index=True)
    st.markdown("**Why it was flagged**")
    for sig in sorted(detail["signal_type"].unique()):
        st.markdown(
            f"- **{ins.label(sig, ins.SIGNAL_LABELS)}**: {ins.SIGNAL_EXPLANATIONS.get(sig, 'No description available.')}"
        )


# ── PAGE 4: FRAUD EXPLORER ───────────────────────────────────────────────


def page_explorer(data: dict, snapshot_id: str):
    s = data["signals"]
    st.title("Fraud explorer")
    explain(
        "What am I looking at?",
        [
            "A signal is one warning raised by one method about one beneficiary. This page lets you filter by "
            "method and signal type to see exactly who was flagged and how strongly.",
            "Rules give fixed scores, so many rows share the same score. Fuzzy matching and Isolation Forest give "
            "graded scores.",
        ],
    )

    c1, c2 = st.columns(2)
    all_methods = sorted(s["method"].unique(), key=lambda m: ins.label(m, ins.METHOD_LABELS))
    all_types = sorted(s["signal_type"].unique(), key=lambda t: ins.label(t, ins.SIGNAL_LABELS))
    methods = c1.multiselect(
        "Method", all_methods, default=all_methods, format_func=lambda m: ins.label(m, ins.METHOD_LABELS)
    )
    types = c2.multiselect(
        "Signal type", all_types, default=all_types, format_func=lambda t: ins.label(t, ins.SIGNAL_LABELS)
    )
    f = s[s["method"].isin(methods) & s["signal_type"].isin(types)]

    with st.expander("What the selected signals mean", expanded=True):
        for t in types:
            st.markdown(
                f"- **{ins.label(t, ins.SIGNAL_LABELS)}**: {ins.SIGNAL_EXPLANATIONS.get(t, 'No description available.')}"
            )

    m1, m2 = st.columns(2)
    m1.metric("Distinct beneficiaries", f"{f['beneficiary_id'].nunique():,}")
    m2.metric("Signal rows", f"{len(f):,}")

    if not f.empty:
        by_type = f.groupby(["method", "signal_type"])["beneficiary_id"].nunique().reset_index(name="n_beneficiaries")
        by_type = ins.display_frame(by_type)
        fig = px.bar(
            by_type,
            x=ins.label("n_beneficiaries"),
            y=ins.label("signal_type"),
            color=ins.label("method"),
            orientation="h",
        )
        whole_number_axis(fig, "x", by_type[ins.label("n_beneficiaries")].max())
        st.plotly_chart(fig)

        st.subheader("Flagged beneficiaries")
        show_table(f.sort_values(["score", "beneficiary_id"], ascending=[False, True]), hide_index=True)
        st.download_button("Download (CSV)", f.to_csv(index=False), "signals_filtered.csv", "text/csv")


# ── PAGE 5: REPORT ───────────────────────────────────────────────────────


def page_report(data: dict, snapshot_id: str):
    st.title("Findings report")
    explain(
        "What am I looking at?",
        [
            "An automatically written report that explains the system, the results and the limitations in plain "
            "language, using the numbers of this snapshot. It is regenerated after every successful pipeline run.",
            "Download it and open it in a browser. Use Print, then Save as PDF, for a PDF copy.",
        ],
    )

    path = SNAPSHOT_ROOT / snapshot_id / "report.html"
    if st.button("Generate (or regenerate) the report for this snapshot"):
        try:
            build_report(SNAPSHOT_ROOT / snapshot_id)
            st.success("Report written to the snapshot folder and to data/reports/.")
        except Exception as exc:
            st.error(f"Could not build the report: {exc}")

    if not path.exists():
        st.info("No report exists for this snapshot yet. Click the button above to create it.")
        return

    html_text = path.read_text(encoding="utf-8")
    if "<section class='technical'>" not in html_text:  # written before report.py used the display names
        st.caption(
            "This report was written before the plain-language labels were added, so it still shows internal "
            "names (for example rule:..., iforest:..., n_flagged). Click Generate above to rebuild it with the "
            "same names as the rest of the dashboard; the numbers do not change."
        )
    st.download_button("Download report (HTML)", html_text, f"fraud_report_{snapshot_id}.html", "text/html")
    try:
        if hasattr(st, "iframe"):  # newer Streamlit (components.html is being retired)
            st.iframe(path, height=900)
        else:
            import streamlit.components.v1 as components

            components.html(html_text, height=900, scrolling=True)
    except Exception:
        st.caption("Preview not available in this Streamlit version; use the download button.")


# ── PAGE 6: GLOSSARY ─────────────────────────────────────────────────────


def page_glossary(data: dict, snapshot_id: str):
    st.title("Glossary")
    st.info(ins.BAG_ANALOGY)
    for term, meaning in ins.GLOSSARY.items():
        st.markdown(f"**{term}**: {meaning}")
    st.subheader("The five kinds of evidence")
    for fam, meaning in ins.FAMILY_EXPLANATIONS.items():
        st.markdown(f"**{ins.label(fam, ins.FAMILY_LABELS)}**: {meaning}")
    st.subheader("The signals")
    for sig, meaning in ins.SIGNAL_EXPLANATIONS.items():
        st.markdown(f"**{ins.label(sig, ins.SIGNAL_LABELS)}**: {meaning}")


PAGES = {
    "Start here": page_start,
    "Overview": page_overview,
    "Method comparison": page_comparison,
    "Investigation queue": page_queue,
    "Fraud explorer": page_explorer,
    "Report": page_report,
    "Glossary": page_glossary,
}


# ── AUTO-REFRESH WATCHER ─────────────────────────────────────────────────


def _watch_for_new_snapshot(loaded_id: str):
    """If publish.py has moved the pointer, rerun the whole app to load the new snapshot."""
    if read_pointer() != loaded_id:
        st.rerun()
    st.caption(f"Checking for new results every {REFRESH_EVERY}")


if hasattr(st, "fragment"):  # Streamlit >= 1.37; older versions just use the Refresh button
    _watch_for_new_snapshot = st.fragment(run_every=REFRESH_EVERY)(_watch_for_new_snapshot)


# ── MAIN ─────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(page_title="Welfare Fraud Detection", layout="wide")

    snapshot_id = read_pointer()
    if snapshot_id is None:
        st.error("No published snapshot yet. Run: python -m src.dashboard.publish --if-layered-run-id <uuid>")
        st.stop()

    try:
        data = load_snapshot(snapshot_id)
    except Exception as exc:  # missing/partial snapshot folder
        st.error(f"Could not read snapshot {snapshot_id}: {exc}")
        st.stop()

    st.sidebar.title("Welfare Fraud Detection")
    page = st.sidebar.radio("Page", list(PAGES))
    st.sidebar.caption(f"Results published {ins.format_ts(data['manifest']['published_at'])}")
    st.sidebar.button("Refresh now")
    with st.sidebar:
        if hasattr(st, "fragment"):
            _watch_for_new_snapshot(snapshot_id)
        else:
            st.caption("Auto-refresh needs Streamlit 1.37 or newer. Click 'Refresh now' to check for new results.")

    PAGES[page](data, snapshot_id)


main()
