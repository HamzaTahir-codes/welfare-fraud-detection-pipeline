# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Welfare-fraud detection pipeline: synthetic data → ETL → PostgreSQL → three detectors (rules, fuzzy record linkage, Isolation Forest) → comparison → dashboard snapshot. Orchestrated by Airflow in Docker; Postgres runs on the host Mac. Design history per stage is in `docs/PROJECT_LOG.md`; Docker/Airflow commands are in `docs/commands.md`.

## Running code

- Run everything from the repo root — paths are relative strings like `data/raw/...`.
- Run modules with `python -m src.<pkg>.<module>`. Exception: `python src/etl/transform_stage.py` must run as a script (it does `from extract_stage import ...`) and runs Extract + Transform together.
- `.env` at the root must define `DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD`; `src/database/config.py` raises if any is missing. Schema: `src/database/schema.sql`.

## Pipeline ordering gotchas

- Stage 1 data generation is manual, not in the DAG, and **not idempotent**: the duplicate/anomaly generators append (including to `data/ground_truth_fraud_ids.csv`) and `inject_messiness.py` rewrites in place. Regenerate from scratch rather than re-running individual scripts. Always pass `--seed 42` (default is None). Order: `generate_beneficiaries` → `generate_national_id` → `generate_disbursements` → `generate_duplicate_ids` → `generate_anomalous_profile` → `inject_messiness` (last).
- `python -m src.etl.loader_stage` TRUNCATES and reloads tables, so `row_id`s change — every load requires re-running all detectors.
- `isolation_forest --layered` must run after `rule_based` and `fuzzy_record` (it excludes entities they already flagged).
- `compare_methods`: get IDs via `--list-runs`, then always pass both `--if-run-id` and `--if-layered-run-id`; defaulting to "latest" can mislabel the layered run as standard.
- `src.dashboard.publish` requires `--if-layered-run-id`; it writes `data/dashboard/<ts>/` and swaps `latest.json` last. The Streamlit app (`src/dashboard/app.py`) only reads the snapshot `latest.json` points to.
- `data/dashboard/latest.json` and the one snapshot it points to are tracked in git so the dashboard can be hosted with no database (deps: `src/dashboard/requirements.txt`). After publishing a new snapshot, update the `!data/dashboard/<id>/` line in `.gitignore`, or the hosted demo breaks. README result numbers come from `data/results/`; keep them in sync.

## Airflow / Docker

- DAG `airflow/dags/welfare_fraud_pipeline.py` must stay compatible with both Airflow 2.10.5 (Docker image) and 3.x (local venv) — keep the `airflow.sdk` / `airflow.decorators` import fallback.
- Never set `DB_HOST` in `docker-compose.yaml`: Airflow's entrypoint reads it and breaks Celery/Redis. Containers get `PROJECT_DB_HOST`, which `_prepare_env()` maps to `DB_HOST`.
- The repo is bind-mounted, so `src/` edits are live in containers; changes to `Dockerfile` or `requirements-airflow.txt` need `docker compose build`. Add new runtime deps to both `requirements.txt` and `requirements-airflow.txt`. Local-only tooling (pytest, ruff, Airflow 3.x for DAG editing) goes in `requirements-dev.txt`.
- `max_active_runs=1` exists because loads truncate shared tables — don't relax it.

## Verification

- `pytest` runs unit tests on hand-built DataFrames — no database or data files needed. Put new pure-function tests in `tests/test_*.py`. Exception: `tests/test_dashboard_pages.py` runs the dashboard headless (Streamlit `AppTest`) against the tracked snapshot (CI installs streamlit and plotly for it; it skips only where they are missing) and builds the findings report from a temporary copy of that snapshot.
- Dashboard display names (column headers, method/signal/pattern names) live in one mapping in `src/dashboard/insights.py` (`COLUMN_LABELS`, `METHOD_LABELS`, ...), applied only at display time by both the dashboard and `report.py`; snapshot CSVs and downloads keep the raw names.
- Lint with `ruff check .` (config in `pyproject.toml`); a hook runs `ruff format` on every edited `.py` file.
- End-to-end, the DAG's `validate_run` task is the quality gate: non-empty tables, fresh result files, and recall/precision floors against the planted answer key (rules 0.99, fuzzy 0.90, layered IF recall 0.60). Changes to detectors should keep those floors.
- `tests/publish_test.py` is a standalone script (`python tests/publish_test.py`) that needs an existing published snapshot; pytest deliberately doesn't collect it.
