---
name: run-pipeline
description: Run the fraud pipeline locally (stages 2–10: ETL, load, detectors, comparison, publish) in the correct order, threading Isolation Forest run IDs through compare and publish. Truncates DB tables.
disable-model-invocation: true
---

Run the local pipeline end to end from the repo root with the venv active. Stop at the first failing step and report its error. `$ARGUMENTS` may contain `--dry-run` (pass it to publish only) or `--skip-publish`.

Preconditions (check, don't fix silently):
- `.env` exists with DB_* vars and Postgres is reachable (`psql "$DB_NAME" -c 'select 1'` or equivalent).
- `data/raw/*.csv` and `data/ground_truth_fraud_ids.csv` exist. If not, stop and tell the user stage 1 generation must be run first (it is not idempotent, so don't run it without asking).
- Warn that the load step TRUNCATES tables; if the Airflow DAG might be running, suggest pausing it first.

Steps:
1. `python src/etl/transform_stage.py` (Extract + Transform; must run as a script)
2. `python -m src.etl.loader_stage`
3. `python -m src.fraud_detection.rule_based`
4. `python -m src.fraud_detection.fuzzy_record`
5. `python -m src.fraud_detection.isolation_forest` — note the run_id it prints
6. `python -m src.fraud_detection.isolation_forest --layered` — note the run_id (must run after 3 and 4)
7. `python -m src.fraud_detection.compare_methods --list-runs` to confirm both IDs, then
   `python -m src.fraud_detection.compare_methods --if-run-id <standard> --if-layered-run-id <layered>`
8. Unless `--skip-publish`: `python -m src.dashboard.publish --if-layered-run-id <layered>` (plus `--dry-run` if given), then `python -m src.dashboard.report`.

Finish with a short summary: row counts loaded, signals per detector, the per-detector precision/recall from `data/results/comparison_detector_metrics.csv` compared with the `validate_run` floors (rules 0.99, fuzzy 0.90, layered IF recall 0.60), and the snapshot path published.
