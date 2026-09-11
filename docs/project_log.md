# Project Log — Multi-Source Data Pipeline & Fraud Detection System

This document tracks build decisions, what was completed at each stage, and why —
kept updated as the project progresses. It doubles as the raw material for the
final report at the end.

---

## Stage 1 — Simulated Source Systems (Data Generation)

**Status:** Complete

### What was built
Three independent synthetic data sources were generated using Python + Faker,
simulating how a real welfare program's data is fragmented across disconnected
government systems:

1. **`generate_beneficiaries.py`** — Beneficiary Registry
   - ~10,000 synthetic beneficiary records with realistic fields: CNIC, name,
     father/husband name, address, phone, bank account, welfare program,
     registration details.
   - Pakistani-format CNIC (`XXXXX-XXXXXXX-X`) and mobile numbers (`03XX-XXXXXXX`)
     used for realism.

2. **`generate_national_id.py`** — National ID / Address Database
   - Reads the beneficiary file and generates a *separate* database representing
     a different government system.
   - ~80% of beneficiaries get a matching record, but with realistic data-entry
     noise: typo'd names, mutated CNIC digits, differently formatted addresses,
     occasional DOB drift.
   - ~20% of beneficiaries are deliberately left with **no matching ID record** —
     simulating identities that don't trace back to a real citizen.
   - Extra "citizen-only" records added (not linked to any beneficiary) to
     simulate a full national population database.

3. **`generate_disbursements.py`** — Bank Disbursement Log
   - Generates recurring monthly payment records for active beneficiaries.
   - **Fraud pattern #1 — Shared account fraud:** a small number of "hub" bank
     accounts are deliberately reused across multiple different beneficiary IDs,
     simulating a single fraudster controlling several fake/duplicate identities.
   - **Fraud pattern #2 — Orphan disbursements:** a small number of payments are
     made to beneficiary IDs that do not exist anywhere in the registry at all.
   - All planted fraud cases are logged separately to
     `data/ground_truth_fraud_ids.csv` — reserved strictly for final evaluation,
     never used by the cleaning or detection code itself.

### Key design decisions
- **Why synthetic data (Faker) instead of real/scraped data:** Real welfare,
  national ID, and bank data is sensitive PII that is never publicly available.
  Synthetic data also allows planting a known ground truth, which is required to
  measure precision/recall of the fraud-detection methods later — something
  impossible with real, unlabeled data.
- **Why CSV files for Stage 1 instead of writing directly to a database:** Mirrors
  how real government data actually arrives — as raw exports/dumps — before an
  ETL process cleans and loads it. Database loading is deliberately scoped to
  Stage 4, not Stage 1.
- **Reproducibility:** all scripts accept a `--seed` argument so the same dataset
  can be regenerated exactly, which matters for consistent grading/demoing.

### Output files produced
- `data/raw/beneficiaries.csv`
- `data/raw/national_id_records.csv`
- `data/raw/disbursements.csv`
- `data/ground_truth_fraud_ids.csv`

---

## Stage 2/3 — Extraction, Staging & Cleaning (ETL)

**Status:** Complete

### What was built

**Stage 2 (Extract):** Processes the three raw CSVs (beneficiaries, disbursements,
national_id_records) from `data/raw/` into `data/staging/`. Implements a
**hub-and-spoke failure model**: beneficiaries failing to extract halts the
entire run (status `FAILED`); disbursements or national_id_records failing
marks the run `PARTIAL` and lets the other source continue. `run_extract()`
returns a structured result dict (status, sources loaded/skipped, run_id) for
downstream stages to consume. Verified end-to-end on real data (~10,100
beneficiaries, ~38,846 disbursements, ~10,000 national ID records).

**Stage 3 (Transform):** Produces three **separately cleaned** DataFrames
(not one merged table), written to `data/clean/`. Locked design decisions:
- **Flag-not-drop policy** — uncleansable/implausible values are never deleted,
  only flagged, preserving full auditability for a fraud-detection context.
- Data quality flags (`dob_parse_failed`, `income_flag`, etc.) and fraud signal
  flags (`shared_bank_account_flag`, `orphaned_beneficiary_flag`) kept in
  **separate columns**, not conflated.
- Dates standardized to ISO `YYYY-MM-DD`; categorical text fields standardized
  to Title Case.
- PKR domain caps applied: 100,000 (monthly income), 10,000 (disbursement
  amount) — values outside range flagged, not dropped.
- **Shared-account fraud signal:** >3 disbursements per bank account flagged.
- **Orphaned disbursement signal:** disbursements referencing a `beneficiary_id`
  absent from the beneficiary registry, flagged via anti-join.
- **CNIC vs. `national_id_no` mismatches** (559 rows) treated as typo errors —
  flagged, not auto-corrected, via a left-join comparison
  (`beneficiary_national_id_joined`).
- Each cleaning run writes **two copies** of every output file: a timestamped
  archive copy (`{source}_cleaned_{run_id}.csv`) and a fixed-name copy
  (`{source}_cleaned.csv`) that downstream stages (loader, and later Airflow)
  read from — avoids fragile filename-guessing in automation.

### Bugs found and resolved
Invalid imports; inverted truthiness checks; numpy `int64` JSON serialization
failures; path separator inconsistencies; inconsistent source naming across
functions; always-true dict key checks; `pd.NA` silently degrading a column
from nullable `Float64` back to `float64`; currency symbols/commas not stripped
from `amount_pkr` before numeric conversion.

### Output files produced
- `data/clean/beneficiaries_cleaned_{run_id}.csv` + `beneficiaries_cleaned.csv`
- `data/clean/national_id_cleaned_{run_id}.csv` + `national_id_cleaned.csv`
- `data/clean/disbursements_cleaned_{run_id}.csv` + `disbursements_cleaned.csv`
- `data/clean/beneficiary_national_id_joined_{run_id}.csv` (CNIC mismatch
  reference join — debug/audit artifact, not used as a source of truth for
  Stage 6 record linkage)

---

## Stage 4 — Database Schema & Loading

**Status:** Complete

### What was built
PostgreSQL schema defined in `src/database/schema.sql` with 5 tables:
`beneficiaries`, `national_id_records`, `disbursements`,
`record_linkage_matches`, `fraud_signals` (the last two are empty output
tables, populated once Stage 6 fraud detection runs).

Loader implemented in `src/etl/loader_stage.py` using `psycopg2` +
`execute_values` for batched inserts. Mirrors Stage 2's hub-and-spoke pattern:
beneficiaries failing to load halts the run; national_id_records or
disbursements failing marks the run `PARTIAL` and lets the other proceed.

### Key design decisions
- **No foreign key constraints** between `beneficiaries`, `disbursements`,
  and `national_id_records`. Orphaned disbursements and CNIC mismatches are
  fraud *signals* to detect, not integrity errors to prevent — an FK
  constraint would make it structurally impossible to load the very fraud
  patterns Stage 1 planted.
- **Surrogate primary keys** (`row_id BIGSERIAL`) used for `beneficiaries` and
  `disbursements` instead of their natural IDs. Both tables contain
  legitimate full-duplicate rows (`is_duplicate_row` flag, 200 rows in
  beneficiaries; ~574 duplicate `disbursement_id`s) that a natural-key
  `PRIMARY KEY` would reject on insert — surrogate keys let every row load
  while `beneficiary_id`/`disbursement_id` remain plain indexed columns for
  grouping/analysis. `national_id_records` has no duplicates, so
  `national_id_no` remains a true primary key.
- **Data quality/fraud flags stored as `TEXT[]`** (native Postgres arrays),
  not comma-separated strings — enables direct queries like
  `'orphaned_beneficiary_flag' = ANY(data_quality_flags)`.
- **Money columns as `NUMERIC(12,2)`**, not float types, to avoid rounding
  errors in any later fraud-amount aggregation.
- Credentials handled via `.env` + `python-dotenv`, loaded through
  `src/database/config.py` and `src/database/connection.py`; never
  hardcoded, `.env` excluded from git.

### Bugs found and resolved
Ambiguous-truth-value crash from calling `pd.isna()` on already-converted list
values (data_quality_flags) — fixed by checking `isinstance(x, list)` before
the NaN check; file-read calls initially outside their per-table `try` blocks,
which broke hub/spoke failure isolation for missing spoke source files;
`sys.path` project-root calculation going up only one directory instead of
two, causing `ModuleNotFoundError: No module named 'src'` — resolved by
running via `python -m src.etl.loader_stage` from the project root instead of
invoking the script directly.

### Verification
Row counts confirmed to match source CSVs exactly (10,100 beneficiaries,
38,846 disbursements). `data_quality_flags` confirmed to load as genuine
Postgres arrays (e.g. `{income_flag,phone_missing}`), not stringified lists.

### Output
- `src/database/schema.sql`
- `src/database/config.py`, `src/database/connection.py`
- `src/etl/loader_stage.py`

---

## Stage 5 — Orchestration

**Status:** ⏳ Not started

*(To be filled in: Airflow DAG design, or fallback approach if Airflow setup was
skipped, and why.)*

---

## Stage 6 — Fraud Detection Experiments & Evaluation

**Status:** ⏳ Not started

*(To be filled in: results of rule-based, fuzzy-linkage, and anomaly detection
methods, with precision/recall/false-positive comparison against ground truth.)*

---

## Stage 7 — Dashboard

**Status:** ⏳ Not started

*(To be filled in: dashboard pages built and key design choices.)*

---

## Open Questions / Follow-ups
- [ ] Confirm with instructor: full Airflow deployment vs. simplified scheduler acceptable?
- [ ] Confirm whether a secondary real public dataset should be blended in for validation.