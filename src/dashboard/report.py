"""
report.py -- Stage 10c: automatic findings report.

Builds ONE self-contained HTML file (no internet, no extra libraries) that
explains the pipeline, the results and the limitations in plain language,
using the numbers in a published snapshot. Because it reads the same snapshot
files as the dashboard, the report and the dashboard always agree.

Open the HTML in a browser; use Print -> Save as PDF for a PDF copy.

Usage (from project root):
    python -m src.dashboard.report                       # newest snapshot
    python -m src.dashboard.report --snapshot-id 20260920143011

Output:
    data/dashboard/<snapshot_id>/report.html
    data/reports/fraud_report_<snapshot_id>.html
    data/reports/fraud_report_latest.html      (fixed name, always the newest)
"""

import argparse
import base64
import html
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from src.dashboard import insights as ins

ROOT = Path(__file__).resolve().parents[2]          # src/dashboard/report.py -> project root
SNAPSHOT_ROOT = ROOT / "data" / "dashboard"
REPORTS_DIR = ROOT / "data" / "reports"

CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1f2937;max-width:960px;margin:0 auto;padding:32px 24px;line-height:1.55}
h1{font-size:28px;margin:0 0 4px}
h2{font-size:21px;margin:36px 0 8px;padding-top:12px;border-top:2px solid #e5e7eb}
h3{font-size:16px;margin:20px 0 6px}
.sub{color:#6b7280;margin-bottom:20px}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}
.tile{flex:1;min-width:160px;background:#f3f4f6;border-radius:10px;padding:12px 14px}
.tile .n{font-size:26px;font-weight:700;color:#1d4ed8}
.tile .l{font-size:13px;color:#4b5563}
.callout{background:#eff6ff;border-left:4px solid #3b82f6;padding:10px 14px;margin:12px 0;border-radius:4px}
.warn{background:#fffbeb;border-left:4px solid #f59e0b;padding:10px 14px;margin:12px 0;border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0}
th{background:#f3f4f6;text-align:left;padding:7px 8px;border-bottom:2px solid #d1d5db}
td{padding:6px 8px;border-bottom:1px solid #e5e7eb;vertical-align:top}
.bar{position:relative;background:#e5e7eb;border-radius:4px;height:18px;min-width:90px}
.bar .fill{background:#3b82f6;height:100%;border-radius:4px}
.bar span{position:absolute;left:8px;top:0;font-size:12px;line-height:18px;color:#111827}
.method{border:1px solid #e5e7eb;border-radius:10px;padding:12px 16px;margin:12px 0}
.small{font-size:13px;color:#6b7280}
img{max-width:100%}
ol li,ul li{margin:4px 0}
@media print{body{max-width:none;padding:0}h2{page-break-after:avoid}.method,table,.tile{page-break-inside:avoid}}
"""


# ── small html helpers ───────────────────────────────────────────────────

def e(x) -> str:
    return html.escape(str(x))


def ul(items) -> str:
    return "<ul>" + "".join(f"<li>{e(i)}</li>" for i in items) + "</ul>"


def bar(v) -> str:
    v = ins.num(v)
    if v is None:
        return "–"
    w = max(0.0, min(1.0, v)) * 100
    return f'<div class="bar"><div class="fill" style="width:{w:.0f}%"></div><span>{w:.0f}%</span></div>'


def df_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, na_rep="–", escape=True,
                      float_format=lambda x: f"{x:.3f}")


def section(num: int, title: str, inner: str) -> str:
    return f"<h2>{num}. {e(title)}</h2>{inner}"


# ── sections ─────────────────────────────────────────────────────────────

def s_summary(data) -> str:
    q, k = data["queue"], data["kpis"]
    validated, _ = ins.split_metrics(data.get("metrics"))
    n_real = sum(1 for _, r in validated.iterrows() if not ins.is_baseline(r.get("detector")))
    tiles = [
        (f"{len(q):,}", "beneficiaries flagged"),
        (f"{int((q['n_families'] >= 2).sum()):,}", "flagged by 2+ kinds of evidence"),
        (f"{int((q['n_validated_families'] >= 2).sum()):,}", "flagged by 2+ validated kinds"),
        (f"{k['not_in_registry']:,}", "IDs not in the registry"),
        (f"{n_real}", "detectors graded vs answer key"),
    ]
    tiles_html = '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="n">{e(n)}</div><div class="l">{e(l)}</div></div>' for n, l in tiles
    ) + "</div>"
    return section(1, "Summary", tiles_html + ul(ins.executive_summary(data)))


def s_system() -> str:
    steps = "".join(f"<li><b>{e(t)}.</b> {e(d)}</li>" for t, d in ins.PIPELINE_STEPS)
    return section(2, "What this system does",
                   f"<p>{e(ins.PROJECT_SUMMARY)}</p><h3>The pipeline, step by step</h3><ol>{steps}</ol>")


def s_data(data) -> str:
    k = data["kpis"]
    counts = pd.DataFrame({"table": list(k["row_counts"].keys()),
                           "rows": [f"{v:,}" for v in k["row_counts"].values()]})
    rows = []
    for table, flags in k["dq_flag_rates"].items():
        for flag, rate in flags.items():
            rows.append({"table": table, "flag": flag,
                         "meaning": ins.DQ_FLAG_EXPLANATIONS.get(flag, ""),
                         "share of rows": f"{rate * 100:.1f}%"})
    inner = (
        "<p>These are the tables in the database when this snapshot was built. Row counts include exact "
        "duplicate rows that were created on purpose as data-quality noise.</p>" + df_html(counts) +
        "<h3>Data quality</h3><p>The source data is deliberately messy, like real government data. The pipeline "
        "flags problems in separate columns and never deletes rows, because a suspicious row may be the fraud "
        "we are looking for.</p>" + df_html(pd.DataFrame(rows))
    )
    return section(3, "The data", inner)


def s_methods(data) -> str:
    sig = pd.DataFrame(data["kpis"]["signals_by_type"])
    blocks = []
    for m in ins.METHODS:
        found = sig[sig["method"] == m["key"]] if not sig.empty else sig
        if found.empty:
            found_html = "<p class='small'>No flags from this method in this snapshot.</p>"
        else:
            found = found.assign(
                signal=found["signal_type"],
                validated=found["signal_type"].isin(ins.VALIDATED_SIGNALS).map({True: "yes", False: "no"}),
            )[["signal", "n_beneficiaries", "validated"]]
            found_html = "<p><b>What it flagged in this run</b></p>" + df_html(found)
        blocks.append(
            f'<div class="method"><h3>{e(m["name"])}</h3>'
            f'<p><b>What it does.</b> {e(m["what"])}</p>'
            f'<p><b>Good at.</b> {e(m["strength"])}</p>'
            f'<p><b>Weak at.</b> {e(m["weakness"])}</p>{found_html}</div>'
        )
    intro = ("<p>Three independent approaches search for fraud. They all write to one shared results table, "
             "which is what makes an honest side-by-side comparison possible.</p>")
    return section(4, "The three detection methods", intro + "".join(blocks))


def s_results(data) -> str:
    validated, heuristic = ins.split_metrics(data.get("metrics"))
    parts = [f'<div class="callout">{e(ins.BAG_ANALOGY)}</div>']

    if validated.empty:
        parts.append("<p class='small'>No validated detector metrics were found in this snapshot.</p>")
    else:
        rows = []
        for _, r in validated.iterrows():
            tp, core = ins.num(r.get("tp_core")), ins.num(r.get("core_size"))
            found = f"{int(tp)} / {int(core)}" if tp is not None and core else "–"
            nf = ins.num(r.get("n_flagged"))
            f1 = ins.num(r.get("f1"))
            rows.append(
                "<tr>"
                f"<td>{e(r.get('detector'))}</td>"
                f"<td>{e(r.get('target', ''))}</td>"
                f"<td>{'' if nf is None else f'{int(nf):,}'}</td>"
                f"<td>{e(found)}</td>"
                f"<td>{bar(r.get('precision'))}</td>"
                f"<td>{bar(r.get('recall'))}</td>"
                f"<td>{'–' if f1 is None else f'{f1:.2f}'}</td>"
                "</tr>"
            )
        parts.append(
            "<h3>Methods graded against the planted answer key</h3>"
            "<table><tr><th>Detector</th><th>Pattern</th><th>Flagged</th><th>Planted cases found</th>"
            "<th>Precision</th><th>Recall</th><th>F1</th></tr>" + "".join(rows) + "</table>"
        )
        parts.append("<h3>In plain words</h3>" +
                     ul([ins.describe_detector(r) for _, r in validated.iterrows()]))

    if not heuristic.empty:
        cols = [c for c in ["detector", "family", "level", "n_flagged", "any_fraud_precision"]
                if c in heuristic.columns]
        parts.append("<h3>Heuristic rules (no answer key, so no precision or recall)</h3>"
                     "<p>These rules are useful leads, but nothing planted lets us grade them. "
                     "'any_fraud_precision' is the share of their flags that happen to belong to any planted pattern.</p>"
                     + df_html(heuristic[cols]))

    if data.get("heatmap_png"):
        b64 = base64.b64encode(Path(data["heatmap_png"]).read_bytes()).decode()
        parts.append("<h3>Which method catches which planted pattern</h3>"
                     "<p>Each cell is the share of that planted pattern the method found. A bright cell means "
                     "the method is good at that pattern; a dark cell means it misses it.</p>"
                     f'<img alt="coverage heatmap" src="data:image/png;base64,{b64}">')
    elif data.get("coverage") is not None:
        parts.append("<h3>Which method catches which planted pattern (recall)</h3>" +
                     data["coverage"].to_html(border=0, na_rep="–", float_format=lambda x: f"{x:.2f}"))

    cols_doc = "".join(f"<li><b>{e(c)}</b>: {e(d)}</li>" for c, d in ins.METRIC_COLUMNS.items())
    parts.append(f"<h3>What the metric names mean</h3><ul>{cols_doc}</ul>")
    return section(5, "How well did each method do?", "".join(parts))


def s_queue(data) -> str:
    q = data["queue"].head(15)
    rows = []
    for _, r in q.iterrows():
        reg = "yes" if bool(r["in_registry"]) else "NOT IN REGISTRY"
        rows.append(
            "<tr>"
            f"<td>{int(r['rank'])}</td><td>{e(r['beneficiary_id'])}</td>"
            f"<td>{int(r['n_validated_families'])} / {int(r['n_families'])}</td>"
            f"<td>{e(ins.why_flagged(r['signal_types']))}</td><td>{e(reg)}</td>"
            "</tr>"
        )
    fam = "".join(f"<li><b>{e(f)}</b>: {e(d)}</li>" for f, d in ins.FAMILY_EXPLANATIONS.items())
    inner = (
        f'<div class="callout">{e(ins.queue_takeaway(data))}</div>'
        "<p>Every flagged beneficiary appears once. Signals that tend to fire for the same reason are grouped "
        "into five evidence families, and scores are never added up, so correlated signals cannot inflate a case. "
        "Beneficiaries are ranked by how many validated families flag them, then by how many families in total, "
        "then by their highest score.</p>"
        f"<h3>The five evidence families</h3><ul>{fam}</ul>"
        "<h3>Top 15 cases</h3>"
        "<table><tr><th>Rank</th><th>Beneficiary</th><th>Validated / all families</th>"
        "<th>Why it was flagged</th><th>In registry?</th></tr>" + "".join(rows) + "</table>"
        "<p class='small'>The full queue is in the dashboard and in case_queue.csv inside the snapshot folder.</p>"
    )
    return section(6, "The investigation queue", inner)


def s_limits(data) -> str:
    return section(7, "Limitations and honest caveats",
                   f'<div class="warn">{ul(ins.limitations(data))}</div>')


def s_glossary() -> str:
    items = "".join(f"<li><b>{e(t)}</b>: {e(d)}</li>" for t, d in ins.GLOSSARY.items())
    return section(8, "Glossary", f"<ul>{items}</ul>")


def s_appendix(data) -> str:
    m = data["manifest"]
    return (
        "<h2>Appendix: run details</h2>"
        "<p class='small'>Identifies exactly which pipeline run produced this report.</p>"
        f"<pre class='small'>{e(json.dumps(m, indent=2, default=str))}</pre>"
    )


# ── build ────────────────────────────────────────────────────────────────

def build_html(data: dict) -> str:
    m = data["manifest"]
    ts = ins.format_ts(m["published_at"])
    body = "".join([
        s_summary(data), s_system(), s_data(data), s_methods(data),
        s_results(data), s_queue(data), s_limits(data), s_glossary(), s_appendix(data),
    ])
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Welfare Fraud Detection - Findings Report</title>"
        f"<style>{CSS}</style></head><body>"
        "<h1>Welfare Fraud Detection: Findings Report</h1>"
        f"<div class='sub'>Snapshot {e(m['snapshot_id'])} &middot; published {e(ts)} &middot; "
        "generated automatically from the pipeline results</div>"
        f"{body}</body></html>"
    )


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy via a temp file in the same folder, then rename over the target.

    copyfile (not copy2) avoids copying file metadata, which fails with
    'Operation not permitted' on Docker bind mounts when the existing target was
    created by a different user (e.g. by you on the host, then overwritten from
    inside the Airflow container). The rename also means a reader never sees a
    half-written file.
    """
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def build_report(snapshot_dir, reports_dir: Path = REPORTS_DIR) -> Path:
    """Write report.html into the snapshot folder and copy it to data/reports/."""
    snapshot_dir = Path(snapshot_dir)
    data = ins.load_snapshot_dir(snapshot_dir)
    html_text = build_html(data)

    out = snapshot_dir / "report.html"
    out.write_text(html_text, encoding="utf-8")

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    snap_id = data["manifest"]["snapshot_id"]

    # Try BOTH copies even if one fails, then say exactly which failed and why.
    errors = []
    for dst in (reports_dir / f"fraud_report_{snap_id}.html", reports_dir / "fraud_report_latest.html"):
        try:
            _atomic_copy(out, dst)
        except OSError as exc:
            errors.append(f"{dst.name}: {exc}")
    if errors:
        raise RuntimeError(f"Report was written to {out} but copying to {reports_dir} failed -> " + "; ".join(errors))
    return out


def main():
    parser = argparse.ArgumentParser(description="Build the automatic findings report (Stage 10c).")
    parser.add_argument("--snapshot-id", default=None, help="Default: the newest snapshot (latest.json).")
    args = parser.parse_args()

    snap_id = args.snapshot_id or json.loads((SNAPSHOT_ROOT / "latest.json").read_text())["snapshot_id"]
    out = build_report(SNAPSHOT_ROOT / snap_id)
    print(f"[report] Wrote {out}")
    print(f"[report] Copies in {REPORTS_DIR}")


if __name__ == "__main__":
    main()