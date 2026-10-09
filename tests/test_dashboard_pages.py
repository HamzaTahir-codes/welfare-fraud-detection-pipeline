"""Smoke and label tests for the Streamlit dashboard (src/dashboard/app.py).

Runs the real app headless with streamlit.testing.v1.AppTest against the
tracked snapshot (data/dashboard/latest.json), so it needs no database.
CI installs streamlit and plotly so the app tests run there too; they skip
only where those are missing. The label and report tests need only pandas.
"""

import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.dashboard import insights as ins

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "dashboard" / "app.py"
SNAPSHOT = ROOT / "data" / "dashboard" / "20260922052809"

PAGES = ["Start here", "Overview", "Method comparison", "Investigation queue", "Fraud explorer", "Report", "Glossary"]

# Internal names that must never reach an investigator as a table header:
# every raw column name the shared mapping relabels (and that is not itself
# a display label), so a regression on any one of them is caught.
RAW_HEADERS = (set(ins.COLUMN_LABELS) | set(ins.METRIC_COLUMN_LABELS)) - (
    set(ins.COLUMN_LABELS.values()) | set(ins.METRIC_COLUMN_LABELS.values())
)


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("plotly")
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_sidebar_lists_every_page(app):
    assert list(app.sidebar.radio[0].options) == PAGES


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [e.message for e in app.exception]
    assert app.title[0].value  # every page draws its heading


@pytest.mark.parametrize("page", ["Method comparison", "Investigation queue", "Fraud explorer"])
def test_tables_show_readable_headers(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert len(app.dataframe) > 0
    for frame in app.dataframe:
        headers = set(map(str, frame.value.columns))
        assert not headers & RAW_HEADERS, (page, headers & RAW_HEADERS)


def _chart_layouts(app, page):
    app.sidebar.radio[0].set_value(page).run()
    return [json.loads(chart.proto.spec)["layout"] for chart in app.get("plotly_chart")]


DEFAULT_TITLES = {
    "value",
    "label",
    "flag",
    "table",
    "metric",
    "method",
    "signal_type",
    "n_beneficiaries",
    "planted_pattern",
    "rate",
    "color",
    "x",
    "y",
}


def _title(layout, key):
    return (layout.get(key, {}).get("title") or {}).get("text")


def test_charts_have_readable_axis_titles(app):
    for page in ["Overview", "Method comparison", "Fraud explorer"]:
        layouts = _chart_layouts(app, page)
        assert layouts, page
        for layout in layouts:
            for axis in ("xaxis", "yaxis"):
                title = _title(layout, axis)
                assert title and title not in DEFAULT_TITLES, (page, axis, title)


def test_charts_have_readable_legend_and_colour_bar_titles(app):
    overview = _chart_layouts(app, "Overview")
    assert [_title(lay, "legend") for lay in overview] == ["Table", "Method", None]

    comparison = _chart_layouts(app, "Method comparison")
    assert _title(comparison[0], "legend") == "Grade"
    heatmap = next(lay for lay in comparison if "coloraxis" in lay)
    assert heatmap["coloraxis"]["colorbar"]["title"]["text"] == "Recall"
    assert heatmap["coloraxis"]["colorbar"]["tickformat"] == ".0%"

    explorer = _chart_layouts(app, "Fraud explorer")
    assert [_title(lay, "legend") for lay in explorer] == ["Method"]


def test_kinds_of_evidence_histogram_has_whole_number_ticks(app):
    layouts = _chart_layouts(app, "Overview")
    hist = next(lay for lay in layouts if lay["xaxis"]["title"]["text"] == "Kinds of evidence")
    assert hist["xaxis"]["dtick"] == 1
    assert hist["xaxis"]["tickformat"] == ",d"


# ── label mapping ────────────────────────────────────────────────────────


def test_label_known_and_unknown_names():
    assert ins.label("n_flagged") == "Beneficiaries flagged"
    assert ins.label("METHOD ENSEMBLE (rules2 + fuzzy + IF)", ins.METHOD_LABELS).startswith("Combined")
    assert ins.label("rule:orphaned_disbursement", ins.METHOD_LABELS) == "Rule: orphaned payment"
    assert ins.label("orphan_disbursement", ins.PATTERN_LABELS) == "Orphaned payments"
    assert ins.label("family", ins.METRIC_COLUMN_LABELS) == "Method type"
    assert ins.label("family") == "Evidence family"
    # unknown names pass through unchanged so a new signal never disappears from view
    assert ins.label("brand_new_signal", ins.SIGNAL_LABELS) == "brand_new_signal"
    assert ins.label("brand_new_column") == "brand_new_column"


def test_label_list_relabels_comma_separated_cells():
    assert ins.label_list("fuzzy_linkage, rule_based", ins.METHOD_LABELS) == "Fuzzy record linkage, Rule-based checks"
    assert ins.label_list(float("nan"), ins.METHOD_LABELS) == ""


def test_every_snapshot_name_has_a_display_label():
    data = ins.load_snapshot_dir(SNAPSHOT)
    metrics, signals, queue = data["metrics"], data["signals"], data["queue"]
    for name in metrics["detector"]:
        assert name in ins.METHOD_LABELS, name
    for name in set(metrics["target"]):
        assert name in ins.PATTERN_LABELS, name
    for name in set(metrics["family"]):
        assert name in ins.METHOD_LABELS, name
    for name in signals["signal_type"].unique():
        assert name in ins.SIGNAL_LABELS and name in ins.SIGNAL_EXPLANATIONS, name
    for name in signals["method"].unique():
        assert name in ins.METHOD_LABELS, name
    for name in signals["family"].unique():
        assert name in ins.FAMILY_LABELS, name
    for name in list(data["coverage"].columns) + list(data["jaccard"].columns) + list(data["unique"]["method"]):
        assert name in ins.METHOD_LABELS, name
    for name in data["coverage"].index:
        assert name in ins.PATTERN_LABELS, name
    for frame in (metrics, signals, queue, data["unique"]):
        for col in frame.columns:
            assert col in ins.COLUMN_LABELS, col
    for flags in data["kpis"]["dq_flag_rates"].values():
        for flag in flags:
            assert flag in ins.DQ_FLAG_LABELS, flag
    for table in data["kpis"]["row_counts"]:
        assert table in ins.TABLE_LABELS, table


def test_display_frame_relabels_without_changing_numbers():
    data = ins.load_snapshot_dir(SNAPSHOT)
    metrics = data["metrics"]
    shown = ins.display_frame(metrics, ins.METRIC_COLUMN_LABELS)
    # the input is untouched, so downloads and computations keep the raw schema
    assert "detector" in metrics.columns
    assert list(shown.columns) == [ins.METRIC_COLUMN_LABELS[c] for c in metrics.columns]
    assert shown["Method type"].iloc[0] == "Rule-based checks"
    assert shown["Detector"].iloc[0] == "Rule: orphaned payment"
    for col in ins.PERCENT_COLUMNS | ins.COUNT_COLUMNS:
        if col in metrics.columns:
            pd.testing.assert_series_equal(shown[ins.METRIC_COLUMN_LABELS[col]], metrics[col], check_names=False)

    # look the case up by id: its rank depends on the ranking, not on the labels
    queue = ins.display_frame(data["queue"])
    case = queue[queue[ins.label("beneficiary_id")] == "BEN-003018"].iloc[0]
    assert case["Methods"] == "Fuzzy record linkage, Rule-based checks"
    assert case["Evidence families"] == "Identity, Payment channel"

    coverage = ins.display_frame(data["coverage"])
    assert coverage.index.name == "Planted pattern"
    assert "Combined (2 validated rules + fuzzy + Isolation Forest), whole method" in coverage.columns
    assert coverage.to_numpy().tolist() == data["coverage"].to_numpy().tolist()

    # the overlap file's first header is "jaccard"; its rows are methods
    jaccard = ins.display_frame(data["jaccard"])
    assert jaccard.index.name == "Method"
    assert "Isolation Forest (layered), whole method" in jaccard.index
    assert "jaccard" not in ins.COLUMN_LABELS


def test_whole_methods_and_single_detectors_have_distinct_labels():
    data = ins.load_snapshot_dir(SNAPSHOT)
    detectors = data["metrics"]["detector"]
    shown = detectors.map(lambda d: ins.label(d, ins.METHOD_LABELS))
    assert shown.is_unique, shown[shown.duplicated()].tolist()
    assert ins.label("iforest:layered", ins.METHOD_LABELS) == "Isolation Forest (layered)"
    assert (
        ins.label("METHOD isolation_forest (layered)", ins.METHOD_LABELS) == "Isolation Forest (layered), whole method"
    )
    assert ins.label("METHOD fuzzy linkage", ins.METHOD_LABELS) == "Fuzzy record linkage, whole method"


def test_display_labels_do_not_merge_distinct_names():
    # two raw names shown under the same label in one table would look like one thing
    data = ins.load_snapshot_dir(SNAPSHOT)
    metrics = data["metrics"]
    validated, heuristic = ins.split_metrics(metrics)
    for frame, col in [
        (data["signals"], "signal_type"),
        (data["signals"], "method"),
        (validated, "detector"),
        (heuristic, "detector"),
        (data["coverage"].reset_index(), "planted_pattern"),
    ]:
        raw = frame[col].drop_duplicates()
        assert raw.map(lambda s, m=ins.VALUE_LABELS[col]: ins.label(s, m)).is_unique, col


def test_plain_words_use_display_names():
    data = ins.load_snapshot_dir(SNAPSHOT)
    validated, _ = ins.split_metrics(data["metrics"])
    text = " ".join(ins.describe_detector(r) for _, r in validated.iterrows())
    assert "rule:" not in text and "iforest:" not in text and "baseline:" not in text
    assert "Isolation Forest (layered) flagged 412 beneficiaries" in text


def test_report_page_flags_a_report_written_with_internal_names(app):
    # the tracked report.html predates the display names, so the page tells investigators to regenerate it
    app.sidebar.radio[0].set_value("Report").run()
    assert not app.exception
    assert any("Click Generate above" in c.value for c in app.caption)


# ── multiselect order ────────────────────────────────────────────────────


@pytest.mark.parametrize("page", ["Investigation queue", "Fraud explorer"])
def test_multiselect_options_are_alphabetical_by_display_label(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert len(app.multiselect) == 2
    for box in app.multiselect:
        shown = list(box.options)
        assert shown == sorted(shown), (page, box.label, shown)


# ── findings report (report.py, embedded in the Report page) ─────────────


@pytest.fixture(scope="module")
def report_html():
    """Build the report from a copy of the tracked snapshot, so the tracked report.html is never rewritten.

    The copy lives in a gitignored folder inside the repo (data/dashboard/* is
    ignored apart from the tracked snapshot) and is removed afterwards."""
    from src.dashboard.report import build_report

    tmp = ROOT / "data" / "dashboard" / "_test_report"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        snap = shutil.copytree(SNAPSHOT, tmp / SNAPSHOT.name)
        out = build_report(snap, reports_dir=tmp / "reports")
        assert out.parent == snap
        assert (tmp / "reports" / "fraud_report_latest.html").exists()
        yield out.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _cells(html_text, tag):
    return [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(rf"<{tag}[^>]*>(.*?)</{tag}>", html_text, re.S)]


def _without_technical(html_text):
    return re.sub(r"<section class='technical'>.*?</section>", "", html_text, flags=re.S)


def test_report_table_headers_are_readable(report_html):
    headers = set(_cells(report_html, "th"))
    assert not headers & RAW_HEADERS, headers & RAW_HEADERS
    for name in ["Detector", "Method type", "Beneficiaries flagged", "Signal", "Table", "Data-quality check"]:
        assert name in headers, name
    assert ins.label("any_fraud_precision", ins.METRIC_COLUMN_LABELS) in headers


def test_report_shows_no_internal_names_outside_the_technical_appendix(report_html):
    assert "<section class='technical'>" in report_html
    body = _without_technical(report_html)
    for raw in ("rule:", "iforest:", "fuzzy:", "baseline:", "METHOD "):
        assert raw not in body, raw
    raw_values = set()
    for mapping in (
        ins.METHOD_LABELS,
        ins.SIGNAL_LABELS,
        ins.PATTERN_LABELS,
        ins.FAMILY_LABELS,
        ins.TABLE_LABELS,
        ins.DQ_FLAG_LABELS,
    ):
        raw_values |= {k for k, v in mapping.items() if k != v}
    cells = set(_cells(body, "td")) | set(_cells(body, "th"))
    assert not cells & raw_values, cells & raw_values


def test_report_tables_and_plain_words_use_the_same_names(report_html):
    cells = _cells(report_html, "td")
    bullets = " ".join(_cells(report_html, "li"))
    for name in ["Rule: orphaned payment", "Isolation Forest (layered)", "Fuzzy matching: duplicate identity"]:
        assert name in cells, name
        assert name in bullets, name
    # heuristic and coverage tables show shares as percentages, not raw ratios
    assert "95.1%" in cells  # fuzzy:duplicate_identity any_fraud_precision 0.9512
    assert "98%" in cells  # fuzzy linkage recall on duplicate_identity 0.975
    assert not any(re.fullmatch(r"0\.\d{3}", c) for c in cells)
