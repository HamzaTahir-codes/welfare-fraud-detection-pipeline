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

**Status:** ⏳ Not started

*(To be filled in once built: what extraction/staging looked like, what cleaning
rules were applied, how fuzzy matching was configured, and what the cleaned
output looked like.)*

---

## Stage 4 — Database Schema & Loading

**Status:** ⏳ Not started

*(To be filled in: final schema design, relationships, any constraints added,
and loading approach.)*

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