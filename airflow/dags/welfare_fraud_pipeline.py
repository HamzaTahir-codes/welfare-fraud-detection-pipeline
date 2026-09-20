"""
welfare_fraud_pipeline.py  --  Stage 9: Orchestration

    extract -> transform -> load -> [ rule_based | fuzzy_linkage | iforest_standard ]
                                          \\             |                /
                                           +--> iforest_layered <--------+
                                                       |
                                               compare_methods -> validate_run (quality gate)
                                                                          |
                                                                 publish_dashboard

Design notes
  * Runs INSIDE the Airflow containers with the project bind-mounted at
    PROJECT_ROOT (default /opt/airflow/project). Every stage keeps its existing
    relative paths (data/raw, data/staging, ...), so each task chdir's to the
    project root first.
  * ETL stages are called in-process (so run_id / status pass between them via
    XCom). Detector stages are run as `python -m ...` subprocesses -- exactly the
    way you run them by hand -- so their argparse entry points are unchanged.
  * loader_stage() TRUNCATES fraud_signals / record_linkage_matches, so the
    detectors must re-run after every load (they do: they are downstream of it)
    and only one DAG run may be active at a time (max_active_runs=1).
  * Isolation Forest --layered excludes entities already flagged by Stage 5/6,
    so it is downstream of all three other detectors. Run IDs for both IF runs are
    captured and handed to compare_methods so the report labels are correct.
  * Stage 1 data generation is deliberately NOT in this DAG: generate_duplicate_identities.py
    appends and inject_messiness.py rewrites in place, so they are not safe to re-run.
  * publish_dashboard runs ONLY if validate_run passed. It writes a snapshot folder (plus the written
    report) under data/dashboard/ and moves latest.json last. The Streamlit app reads only that
    snapshot, so a failed or in-progress run never changes what the dashboard shows.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pendulum

try:                                   # Airflow 3.x
    from airflow.sdk import dag, task
except ImportError:                    # Airflow 2.x
    from airflow.decorators import dag, task

# ── CONFIG ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/opt/airflow/project"))

# ASSUMPTION: adjust if your module paths differ.
MODULE_RULE_BASED = "src.fraud_detection.rule_based"
MODULE_FUZZY = "src.fraud_detection.fuzzy_record"
MODULE_IFOREST = "src.fraud_detection.isolation_forest"
MODULE_COMPARE = "src.fraud_detection.compare_methods"
MODULE_PUBLISH = "src.dashboard.publish"
IFOREST_LAYERED_FLAG = "--layered"

# Schedule: cron evaluated in TIMEZONE. Docker on a laptop only runs while the laptop is awake,
# so pick a time it is usually on. Set to None to go back to manual-only.
SCHEDULE_CRON = "0 21 * * *"          # daily 21:00
TIMEZONE = "Asia/Karachi"

# Tables that must be non-empty after a run.
REQUIRED_TABLES = ["beneficiaries", "national_id_records", "disbursements",
                   "fraud_signals", "record_linkage_matches"]

# Regression floors on PLANTED patterns (read from data/results/comparison_detector_metrics.csv).
# These catch silent breakage; they are not claims about real-world fraud detection.
QUALITY_GATES = {
    "rule:orphaned_disbursement": {"recall": 0.99, "precision": 0.99},
    "rule:shared_bank_account":   {"recall": 0.99, "precision": 0.99},
    "fuzzy:duplicate_identity":   {"recall": 0.99, "precision": 0.99},
    "iforest:layered":            {"recall": 0.60},
}

# Hub-and-spoke semantics: PARTIAL load (a spoke table failed) continues with a
# warning by default. Set True to make PARTIAL fail the run instead.
FAIL_ON_PARTIAL_LOAD = False


# ── HELPERS ───────────────────────────────────────────────────────────────

def _prepare_env() -> None:
    """Make the project's relative paths and imports behave as they do from your shell."""
    os.chdir(PROJECT_ROOT)
    # Airflow's own entrypoint reads DB_HOST, so the containers get PROJECT_DB_HOST and it is mapped here
    if os.environ.get("PROJECT_DB_HOST"):
        os.environ["DB_HOST"] = os.environ["PROJECT_DB_HOST"]
    for sub in ("data/staging", "data/clean", "data/results"):
        (PROJECT_ROOT / sub).mkdir(parents=True, exist_ok=True)
    # project root for `src.*` imports; src/etl for transform_stage's `from extract_stage import ...`
    for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src" / "etl")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _run_module(module: str, *args: str) -> None:
    """Run `python -m <module> <args>` from the project root, streaming output into the task log."""
    _prepare_env()   # chdir + PROJECT_DB_HOST -> DB_HOST, BEFORE the subprocess env is copied
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PROJECT_ROOT), env.get("PYTHONPATH", "")]))
    cmd = [sys.executable, "-m", module, *args]
    print(f"$ cd {PROJECT_ROOT} && {' '.join(cmd)}", flush=True)
    with subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, text=True, bufsize=1,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{module} exited with code {proc.returncode}")


def _latest_iforest_run_id() -> str:
    """run_id of the most recently written Isolation Forest run in fraud_signals."""
    _prepare_env()
    from src.database.connection import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT run_id FROM fraud_signals WHERE method LIKE 'isolation_forest%' "
                "ORDER BY created_at DESC LIMIT 1;"
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise RuntimeError("Isolation Forest finished but wrote no rows to fraud_signals.")
    return str(row[0])


# ── DAG ───────────────────────────────────────────────────────────────────

@dag(
    dag_id="welfare_fraud_pipeline",
    description="ETL -> rule-based / fuzzy / Isolation Forest detection -> comparative evaluation",
    schedule=SCHEDULE_CRON,
    start_date=pendulum.datetime(2026, 9, 1, tz=TIMEZONE),
    catchup=False,                 # never backfill missed days
    max_active_runs=1,             # loader truncates shared tables: never run two at once
    dagrun_timeout=timedelta(hours=2),
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["welfare-fraud", "etl", "fraud-detection"],
    doc_md=__doc__,
)
def welfare_fraud_pipeline():

    # ---------- Stage 2: Extract (hub-and-spoke) ----------
    @task(task_id="extract")
    def extract() -> dict:
        _prepare_env()
        from src.etl.extract_stage import run_extract

        summary = run_extract()
        if summary["status"] == "FAILED":
            raise RuntimeError(f"Extract failed: {summary.get('message')}")
        return {"run_id": summary["run_id"], "status": summary["status"],
                "skipped_sources": summary.get("skipped_sources", [])}

    # ---------- Stage 3: Transform ----------
    @task(task_id="transform")
    def transform(extract_summary: dict) -> dict:
        _prepare_env()
        from src.etl.transform_stage import run_transform_stage

        result = run_transform_stage(extract_summary)
        if str(result["status"]).lower() == "failed":
            raise RuntimeError(f"Transform failed: {result.get('reason')}")
        return {"run_id": result["run_id"], "status": result["status"],
                "skipped_sources": result.get("skipped_sources", [])}

    # ---------- Stage 4: Load (idempotent: truncates first) ----------
    @task(task_id="load")
    def load(transform_result: dict) -> dict:
        _prepare_env()
        from src.etl.loader_stage import loader_stage

        result = loader_stage()
        print(result, flush=True)
        if result["status"] == "FAILED":
            raise RuntimeError(f"Load failed: {result.get('errors')}")
        if result["status"] == "PARTIAL":
            msg = f"Load PARTIAL - tables failed: {result.get('tables_failed')} {result.get('errors')}"
            if FAIL_ON_PARTIAL_LOAD:
                raise RuntimeError(msg)
            print(f"[WARN] {msg}", flush=True)
        return {"status": result["status"], "tables_loaded": result["tables_loaded"]}

    # ---------- Stage 5 ----------
    @task(task_id="rule_based")
    def rule_based(_load: dict) -> None:
        _run_module(MODULE_RULE_BASED)

    # ---------- Stage 6 ----------
    @task(task_id="fuzzy_linkage")
    def fuzzy_linkage(_load: dict) -> None:
        _run_module(MODULE_FUZZY)

    # ---------- Stage 7 (standard) ----------
    @task(task_id="iforest_standard")
    def iforest_standard(_load: dict) -> str:
        _run_module(MODULE_IFOREST)
        run_id = _latest_iforest_run_id()
        print(f"Isolation Forest (standard) run_id = {run_id}", flush=True)
        return run_id

    # ---------- Stage 7 (layered: needs Stage 5/6 output in the DB) ----------
    @task(task_id="iforest_layered")
    def iforest_layered() -> str:
        _run_module(MODULE_IFOREST, IFOREST_LAYERED_FLAG)
        run_id = _latest_iforest_run_id()
        print(f"Isolation Forest (layered) run_id = {run_id}", flush=True)
        return run_id

    # ---------- Stage 8 ----------
    @task(task_id="compare_methods")
    def compare_methods(standard_run_id: str, layered_run_id: str) -> None:
        if standard_run_id == layered_run_id:
            raise RuntimeError("Standard and layered Isolation Forest run_ids are identical - "
                               "check that --layered actually created a second run.")
        _run_module(MODULE_COMPARE, "--if-run-id", standard_run_id,
                    "--if-layered-run-id", layered_run_id)

    # ---------- Quality gate: did the run actually produce sane results? ----------
    @task(task_id="validate_run")
    def validate_run(**context) -> dict:
        _prepare_env()
        import pandas as pd
        from src.database.connection import get_connection

        failures, report = [], []
        run_start = context["dag_run"].start_date

        # 1. every table has rows
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                for table in REQUIRED_TABLES:
                    cur.execute(f"SELECT COUNT(*) FROM {table};")
                    n = cur.fetchone()[0]
                    report.append(f"rows[{table}] = {n}")
                    if n == 0:
                        failures.append(f"table {table} is empty")
        finally:
            conn.close()

        # 2. result files were written by THIS run (not left over from an earlier one)
        results_dir = PROJECT_ROOT / "data" / "results"
        for name in ("comparison_detector_metrics.csv", "comparison_report.md"):
            f = results_dir / name
            if not f.exists():
                failures.append(f"{name} missing")
                continue
            mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
            report.append(f"{name} written {mtime:%Y-%m-%d %H:%M:%S} UTC")
            if run_start and mtime < run_start:
                failures.append(f"{name} is stale (older than this DAG run)")

        # 3. detector quality gates against the planted answer key
        metrics_path = results_dir / "comparison_detector_metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path).set_index("detector")
            for detector, floors in QUALITY_GATES.items():
                if detector not in metrics.index:
                    failures.append(f"{detector} missing from comparison metrics")
                    continue
                for metric, floor in floors.items():
                    value = metrics.loc[detector, metric]
                    ok = bool(pd.notna(value) and value >= floor)
                    report.append(f"{'PASS' if ok else 'FAIL'}  {detector} {metric} = {value:.3f} (floor {floor})")
                    if not ok:
                        failures.append(f"{detector} {metric} = {value} below floor {floor}")

        print("\n".join(report), flush=True)
        if failures:
            raise RuntimeError("Validation failed:\n  - " + "\n  - ".join(failures))
        return {"checks_passed": len(report)}

    # ---------- Stage 10: publish the dashboard snapshot + written report ----------
    @task(task_id="publish_dashboard")
    def publish_dashboard(layered_run_id: str, **context) -> None:
        # Downstream of validate_run, so it only fires for a run that passed the quality gate.
        # The layered Isolation Forest run_id arrives by XCom, the same way compare_methods gets it.
        _run_module(MODULE_PUBLISH, "--if-layered-run-id", layered_run_id,
                    "--dag-run-id", context["dag_run"].run_id)

    # ---------- wiring ----------
    loaded = load(transform(extract()))
    rb = rule_based(loaded)
    fz = fuzzy_linkage(loaded)
    if_std = iforest_standard(loaded)
    if_lay = iforest_layered()
    [rb, fz, if_std] >> if_lay
    cmp = compare_methods(if_std, if_lay)
    val = validate_run()
    pub = publish_dashboard(if_lay)
    cmp >> val >> pub


welfare_fraud_pipeline()