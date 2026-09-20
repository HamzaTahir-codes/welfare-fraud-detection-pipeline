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

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]          # src/dashboard/app.py -> project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard import insights as ins            # noqa: E402
from src.dashboard.report import build_report        # noqa: E402

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
    explain("What am I looking at?", [
        "This page is the health check of one pipeline run: how much data went in, how messy it was, "
        "and how many beneficiaries each method flagged.",
        "The numbers come from a snapshot, a frozen copy of the results of the last successful run.",
    ])
    st.info(ins.queue_takeaway(data))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Snapshot", m["snapshot_id"])
    c2.metric("Published", ins.format_ts(m["published_at"]))
    c3.metric("Beneficiaries in queue", f"{k['queue_size']:,}")
    c4.metric("IDs not in registry", f"{k['not_in_registry']:,}")
    st.caption("Snapshot ids are UTC timestamps (year, month, day, hour, minute, second). "
               "The Published time is shown in Pakistan time.")
    with st.expander("Run details (which pipeline run produced this)"):
        st.json({"dag_run_id": m.get("dag_run_id"), "run_ids": m.get("run_ids")})

    st.subheader("Rows in the database when this snapshot was built")
    st.caption("Counts include exact duplicate rows that were created on purpose as data-quality noise.")
    cols = st.columns(len(k["row_counts"]))
    for col, (table, n) in zip(cols, k["row_counts"].items()):
        col.metric(table.replace("_", " "), f"{n:,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Data-quality flag rates")
        st.caption("The source data is deliberately messy, like real government data. Problems are flagged, never deleted.")
        rows = [{"table": t, "flag": f, "rate": v}
                for t, flags in k["dq_flag_rates"].items() for f, v in flags.items()]
        fig = px.bar(pd.DataFrame(rows), x="rate", y="flag", color="table",
                     orientation="h", barmode="group", text_auto=".1%")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig)
        with st.expander("What each flag means"):
            for flag, meaning in ins.DQ_FLAG_EXPLANATIONS.items():
                st.markdown(f"- **{flag}**: {meaning}")
    with right:
        st.subheader("Beneficiaries flagged, by signal")
        st.caption("Each bar is the number of distinct beneficiaries one signal flagged. The same person can appear in several bars.")
        sig = pd.DataFrame(k["signals_by_type"])
        fig = px.bar(sig, x="n_beneficiaries", y="signal_type", color="method", orientation="h")
        st.plotly_chart(fig)

    st.subheader("How many kinds of evidence flag each beneficiary?")
    st.caption("One kind of evidence is a weak lead. Two or more independent kinds is much stronger.")
    fam = pd.DataFrame({"kinds of evidence": list(k["queue_by_families"].keys()),
                        "beneficiaries": list(k["queue_by_families"].values())})
    fig = px.bar(fam, x="kinds of evidence", y="beneficiaries", text_auto=True)
    st.plotly_chart(fig)


# ── PAGE 2: METHOD COMPARISON ────────────────────────────────────────────

def _pick(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]]


def page_comparison(data: dict, snapshot_id: str):
    st.title("Method comparison")
    explain("What am I looking at?", [
        "We planted known fraud in the data, so we can grade every method honestly. This page shows how "
        "each method did. Precision and recall are the two grades.",
        ins.BAG_ANALOGY,
    ])

    metrics = data["metrics"]
    if metrics is None:
        st.warning("comparison_detector_metrics.csv was not found in this snapshot.")
        return
    validated, heuristic = ins.split_metrics(metrics)

    st.subheader("Graded against the planted answer key")
    st.dataframe(
        _pick(validated, ["detector", "family", "level", "target", "n_flagged", "tp_core",
                          "core_size", "precision", "recall", "f1", "precision_strict",
                          "any_fraud_precision", "overall_core_recall"]),
        hide_index=True,
    )
    with st.expander("What each column means"):
        for col, meaning in ins.METRIC_COLUMNS.items():
            st.markdown(f"- **{col}**: {meaning}")

    if {"precision", "recall", "f1"} <= set(validated.columns) and not validated.empty:
        plot = validated.copy()
        label = plot["detector"].astype(str)
        if "target" in plot.columns:
            label = label + " / " + plot["target"].astype(str)
        plot["label"] = label
        long = plot.melt(id_vars="label", value_vars=["precision", "recall", "f1"],
                         var_name="metric", value_name="value")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        fig = px.bar(long, x="label", y="value", color="metric", barmode="group", range_y=[0, 1])
        st.plotly_chart(fig)

    if not validated.empty:
        st.subheader("In plain words")
        for _, r in validated.iterrows():
            st.markdown(f"- {ins.describe_detector(r)}")

    st.subheader("Heuristic rules (no answer key, so no precision or recall)")
    st.caption("These are useful leads, but nothing planted lets us grade them.")
    st.dataframe(
        _pick(heuristic, ["detector", "family", "level", "n_flagged", "any_fraud_precision"]),
        hide_index=True,
    )

    st.subheader("Which method catches which planted pattern (recall)")
    st.caption("Each cell is the share of that planted pattern the method found. Bright means it finds it, dark means it misses it.")
    coverage = data["coverage"]
    shown = False
    if coverage is not None:
        try:
            matrix = coverage.apply(pd.to_numeric, errors="coerce")
            fig = px.imshow(matrix, text_auto=".2f", aspect="auto", zmin=0, zmax=1,
                            color_continuous_scale="Blues")
            st.plotly_chart(fig)
            shown = True
        except Exception:
            shown = False
    if not shown and data["heatmap_png"]:
        st.image(data["heatmap_png"])

    with st.expander("Unique catches and overlap between methods"):
        if data["unique"] is not None:
            st.caption("Entities caught by only one method")
            st.dataframe(data["unique"])
        if data["jaccard"] is not None:
            st.caption("Overlap (Jaccard similarity) between methods: 0 means no shared catches, 1 means identical.")
            st.dataframe(data["jaccard"])

    st.subheader("How to read these results honestly")
    for line in ins.limitations(data):
        st.markdown(f"- {line}")

    if data["report_md"]:
        with st.expander("Full comparison report (from the pipeline)"):
            st.markdown(data["report_md"])


# ── PAGE 3: INVESTIGATION QUEUE ──────────────────────────────────────────

def page_queue(data: dict, snapshot_id: str):
    q = data["queue"]
    st.title("Investigation queue")
    explain("What am I looking at?", [
        "This is the list of people an investigator should look at, most suspicious first. Every flagged "
        "beneficiary appears once.",
        "Signals that tend to fire for the same reason are grouped into five kinds of evidence (families), "
        "and scores are never added up, so correlated signals cannot inflate a case.",
        "Ranking: first by how many validated kinds of evidence flag the person, then by how many kinds in "
        "total, then by their highest score.",
    ])
    st.info(ins.queue_takeaway(data))
    with st.expander("The five kinds of evidence"):
        for fam, meaning in ins.FAMILY_EXPLANATIONS.items():
            st.markdown(f"- **{fam}**: {meaning}")

    all_families = sorted({f.strip() for s in q["families"] for f in str(s).split(",")})
    all_methods = sorted({m.strip() for s in q["methods"] for m in str(s).split(",")})

    c1, c2, c3, c4 = st.columns(4)
    min_val = c1.slider("Min validated families", 0, max(int(q["n_validated_families"].max()), 1), 0)
    min_fam = c2.slider("Min total families", 1, max(int(q["n_families"].max()), 2), 1)
    need_fams = c3.multiselect("Must include family", all_families)
    need_methods = c4.multiselect("Must include method", all_methods)
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
    st.dataframe(
        view,
        hide_index=True,
        column_config={
            "max_score": st.column_config.ProgressColumn("max_score", min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )
    st.download_button("Download filtered queue (CSV)", view.to_csv(index=False),
                       "case_queue_filtered.csv", "text/csv")

    st.subheader("Case detail")
    options = view["beneficiary_id"].head(500).tolist()
    if not options:
        st.caption("No beneficiaries match the filters.")
        return
    chosen = st.selectbox("Beneficiary (top 500 of the current filter)", options)
    row = view[view["beneficiary_id"] == chosen].iloc[0]
    if not row["in_registry"]:
        st.warning("This ID is not in the beneficiary registry: a payment was made against an ID that was never registered.")

    detail = data["signals"][data["signals"]["beneficiary_id"] == chosen]
    st.dataframe(detail[["method", "signal_type", "family", "score", "n_flags"]], hide_index=True)
    st.markdown("**Why it was flagged**")
    for sig in sorted(detail["signal_type"].unique()):
        st.markdown(f"- **{sig}**: {ins.SIGNAL_EXPLANATIONS.get(sig, 'No description available.')}")


# ── PAGE 4: FRAUD EXPLORER ───────────────────────────────────────────────

def page_explorer(data: dict, snapshot_id: str):
    s = data["signals"]
    st.title("Fraud explorer")
    explain("What am I looking at?", [
        "A signal is one warning raised by one method about one beneficiary. This page lets you filter by "
        "method and signal type to see exactly who was flagged and how strongly.",
        "Rules give fixed scores, so many rows share the same score. Fuzzy matching and Isolation Forest give "
        "graded scores.",
    ])

    c1, c2 = st.columns(2)
    methods = c1.multiselect("Method", sorted(s["method"].unique()), default=sorted(s["method"].unique()))
    types = c2.multiselect("Signal type", sorted(s["signal_type"].unique()),
                           default=sorted(s["signal_type"].unique()))
    f = s[s["method"].isin(methods) & s["signal_type"].isin(types)]

    with st.expander("What the selected signals mean", expanded=True):
        for t in types:
            st.markdown(f"- **{t}**: {ins.SIGNAL_EXPLANATIONS.get(t, 'No description available.')}")

    m1, m2 = st.columns(2)
    m1.metric("Distinct beneficiaries", f"{f['beneficiary_id'].nunique():,}")
    m2.metric("Signal rows", f"{len(f):,}")

    if not f.empty:
        by_type = (f.groupby(["method", "signal_type"])["beneficiary_id"].nunique()
                   .reset_index(name="beneficiaries"))
        fig = px.bar(by_type, x="beneficiaries", y="signal_type", color="method", orientation="h")
        st.plotly_chart(fig)

        st.subheader("Flagged beneficiaries")
        st.dataframe(f.sort_values(["score", "beneficiary_id"], ascending=[False, True]), hide_index=True)
        st.download_button("Download (CSV)", f.to_csv(index=False), "signals_filtered.csv", "text/csv")


# ── PAGE 5: REPORT ───────────────────────────────────────────────────────

def page_report(data: dict, snapshot_id: str):
    st.title("Findings report")
    explain("What am I looking at?", [
        "An automatically written report that explains the system, the results and the limitations in plain "
        "language, using the numbers of this snapshot. It is regenerated after every successful pipeline run.",
        "Download it and open it in a browser. Use Print, then Save as PDF, for a PDF copy.",
    ])

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
    st.download_button("Download report (HTML)", html_text, f"fraud_report_{snapshot_id}.html", "text/html")
    try:
        if hasattr(st, "iframe"):            # newer Streamlit (components.html is being retired)
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
        st.markdown(f"**{fam}**: {meaning}")
    st.subheader("The signals")
    for sig, meaning in ins.SIGNAL_EXPLANATIONS.items():
        st.markdown(f"**{sig}**: {meaning}")


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


if hasattr(st, "fragment"):   # Streamlit >= 1.37; older versions just use the Refresh button
    _watch_for_new_snapshot = st.fragment(run_every=REFRESH_EVERY)(_watch_for_new_snapshot)


# ── MAIN ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Welfare Fraud Detection", layout="wide")

    snapshot_id = read_pointer()
    if snapshot_id is None:
        st.error("No published snapshot yet. Run: "
                 "python -m src.dashboard.publish --if-layered-run-id <uuid>")
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