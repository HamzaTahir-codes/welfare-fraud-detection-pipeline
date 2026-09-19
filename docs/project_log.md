# Project Log — Multi-Source Data Pipeline & Fraud Detection System

This document tracks build decisions, what was completed at each stage, and why —
kept updated as the project progresses. It doubles as the raw material for the
final report at the end.

---

## Status at a glance (updated 2026-09-19)

| Stage | What | Status |
|---|---|---|
| 1 | Simulated source systems + planted fraud patterns | Complete (extended in Stages 6 and 7) |
| 2/3 | Extract, staging, cleaning | Complete |
| 4 | PostgreSQL schema and loader | Complete (loader made idempotent) |
| 5 | Rule-based detection | Complete, validated on 2 of 6 rules |
| 6 | Fuzzy record linkage (duplicate identities) | Complete, validated |
| 7 | Isolation Forest (multivariate anomalies) | Complete, validated with limitations |
| 8 | Comparative evaluation of the three methods | Not started |
| 9 | Orchestration (Airflow or fallback) | Not started |
| 10 | Dashboard | Not started |

**Numbering note:** the original plan labelled orchestration, experiments and the dashboard as
Stages 5-7. Rule-based detection, fuzzy linkage and Isolation Forest are now Stages 5-7, so those
three items moved to Stages 8-10.

**Dataset versions:** the Stage 1-4 entries below were written against the first dataset
(~10,100 beneficiaries, ~38,846 disbursements). For the Stage 6-7 work the data was regenerated
smaller, and the current dataset is: 5,090 beneficiary rows, 18,844 disbursement rows, 6,040
national ID rows, with a ground-truth answer key of 390 entries (25 `orphan_disbursement`,
300 `shared_account`, 40 `duplicate_identity`, 25 `multivariate_anomaly`). Every number
below is labelled with the dataset it came from.

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

### Later extensions (planted patterns added for Stages 6 and 7)
- **`generate_duplicate_identities.py`** plants `duplicate_identity` ground truth (see Stage 6).
- **`generate_anomalous_profiles.py`** plants `multivariate_anomaly` ground truth (see Stage 7).
- Run order: the three base generators -> `generate_duplicate_identities.py` ->
  `generate_anomalous_profiles.py` -> `inject_messiness.py`.

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

### Later bug found during Stage 7: "Rs." amounts parsed as tiny numbers
`strip_currency_symbols` removed the letters of "Rs." but left the dot, so `"Rs. 10,000"` became
`.10000` and was parsed as 0.1 (likewise 0.5, 0.6, 0.75, 0.8). About 8% of `amount_pkr_clean`
values (1,483 of 18,536 rows in the cleaned file at the time) were wrong. It was not caught earlier
because none of the Stage 5/6 rules use payment amounts. Fixed in `strip_currency_symbols`. Confirmed after the ETL re-run: the Stage 7 script, which raises a warning for any amount
under 1,000, printed no warning.

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

### Later update: loader is idempotent
Adding new planted patterns meant the raw data had to be regenerated and reloaded. Dropping and
recreating the database each time was rejected. Instead `loader_stage()` now truncates
`beneficiaries`, `national_id_records`, `disbursements`, `fraud_signals` and `record_linkage_matches`
(`RESTART IDENTITY`) at the start of every run. `fraud_signals` and `record_linkage_matches` are
included because they key off `row_id`, which changes on any reload. **Stages 5, 6 and 7 must be re-run after
every loader run.**

## Stage 5 — Rule-Based Fraud Detection Baseline (2026-09-14)

### What was built
Implemented `src/fraud_detection/rule_based.py` — six rule-based fraud
detectors (duplicate CNIC across identities, shared bank account,
orphaned disbursement, CNIC mismatch, duplicate-cycle payment, payment
to inactive/suspended beneficiaries), each reading from Postgres into
pandas, applying vectorized rule logic, and bulk-writing results to
`fraud_signals` via `execute_values`. Also built
`src/fraud_detection/evaluate_rule_based.py`, which scores signals
against the planted ground truth in `data/ground_truth_fraud_ids.csv`.

### Bugs found and fixed during evaluation

**1. `shared_bank_account` was flagging 99.5% of all disbursements.**
Root cause: the original Stage 3 flag counted *total disbursement rows
per account* (>3), which any legitimate beneficiary paid monthly across
6 cycles naturally exceeds. Fixed by recomputing the signal in Stage 5
as *distinct beneficiary_id count per account* (>1) — the actual
fraud pattern from `plant_shared_account_fraud`. Signal count dropped
from 38,664 to 5,874. Decision: keep the old Stage 3 column as legacy/
unused rather than re-running Stage 3+4, since this is fraud-detection
logic and belongs in Stage 5 conceptually.

**2. `entity_id` used business identifiers instead of the surrogate key.**
`inject_messiness.py` duplicates full rows (same `disbursement_id`/
`beneficiary_id` on both copies), so business IDs aren't guaranteed
unique per physical row — the same principle already established for
`row_id` vs. natural keys in Stage 4. Fixed: `entity_id` in
`fraud_signals` now always stores `row_id` (cast to text), not the
business ID. **This supersedes the earlier locked decision** that
`entity_id` should use business identifiers.

**3. `duplicate_cycle_payment` was partly counting messiness artifacts,
not fraud.** `inject_messiness.py`'s row-duplication (~1.5% of rows)
creates exact-copy rows sharing the same `disbursement_id`. Fixed by
collapsing rows to one event per distinct `disbursement_id` before
checking for genuine same-cycle duplicates. Signal count dropped from
6,164 to 5,590.

### Validated results (against ground truth: 25 orphan, 910 shared_account)

| Signal | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| `orphaned_disbursement` | 1.0000 | 1.0000 | 1.0000 | Exact match, 25/25 |
| `shared_bank_account` | 0.8465 | 1.0000 | 0.9169 | All 165 "FPs" are hub-account owners not logged as victims in ground truth — effectively 100% precision once accounted for |

### Key finding: `duplicate_cycle_payment` is not an independent signal
Overlap analysis vs. `shared_bank_account`: 51.4% of duplicate-cycle
signals coincide with shared-account signals, and the ~50% split is
structurally expected — each victim's fraud pair produces one flagged
row (hub account) and one unflagged row (their own account). Conclusion:
`duplicate_cycle_payment` mostly re-detects the same shared-account
fraud mechanism from a different angle, not a distinct fraud pattern.
**Decision: do not treat these two signals as independent evidence when
aggregating a combined risk score later** (Stage 6/dashboard) — double-
counting one underlying case would overstate confidence.

### Unverified rules (no planted ground truth to score against)
`cnic_mismatch` (556 flagged), `payment_to_inactive_beneficiary` (1,824
flagged), `duplicate_cnic_multiple_identities` (0 flagged) — plausible
heuristics, but `generate_disbursements.py`/`generate_beneficiaries.py`
don't plant corresponding ground truth. Report these as unverified
heuristic signal counts, not precision/recall.

### Re-run on the current dataset (after Stage 6-7 data regeneration)
Signal counts: `shared_bank_account` 1,992; `duplicate_cycle_payment` 2,009;
`payment_to_inactive_beneficiary` 596; `cnic_mismatch` 326; `orphaned_disbursement` 25;
`duplicate_cnic_multiple_identities` 0.

| Signal | TP / FP / FN | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|
| `orphaned_disbursement` | 25 / 0 / 0 | 1.0000 | 1.0000 | 1.0000 | Exact match |
| `shared_bank_account` (beneficiary level) | 300 / 50 / 0 | 0.8571 | 1.0000 | 0.9231 | All 50 FPs are hub-account owners not logged as victims in the answer key |

`shared_bank_account` vs `duplicate_cycle_payment`: 1,021 overlapping row_ids, 50.8% of the
duplicate-cycle signals, Jaccard 0.3426. This is consistent with the earlier finding that they are not
independent evidence.

### Status
Rule-based baseline complete and validated (2 of 6 rules have ground truth). Next: fuzzy
record linkage.

---

## Stage 6 — Fuzzy Record Linkage: Duplicate-Identity Ghost Beneficiaries (2026-09-15)

**Status:** Complete, validated

### What was built
- **`generate_duplicate_identities.py`** (Stage 1 extension): plants 40 `duplicate_identity` pairs. Each
  duplicate is the same person registered again under a new `beneficiary_id`, with a typo in the name,
  a reworded address, the same date of birth, a fabricated CNIC and bank account, and its own disbursement
  history. Logged to `ground_truth_fraud_ids.csv` (`original_beneficiary_id`, `duplicate_beneficiary_id`, one `run_id`
  per execution).
- **`src/fraud_detection/fuzzy_record.py`**: self-linkage of the `beneficiaries` table, writing pairs to
  `record_linkage_matches`.

### Key design decisions
- **Scope:** beneficiaries against beneficiaries, not beneficiaries against `national_id_records`, because the
  latter would mostly re-detect the exact-match `cnic_mismatch` signal.
- **Blocking on exact `date_of_birth_clean`:** planted duplicates keep the DOB unchanged, so no true pair is
  blocked out, while comparisons drop from O(n^2) to same-birthday groups (547 candidate pairs on the current dataset).
- **Scoring:** RapidFuzz `token_sort_ratio` on `full_name` (weight 0.6) and `address_line` (0.4), after stripping
  domain stopwords (honorifics such as "Muhammad", "Bibi", "Khan"; address boilerplate such as "House", "Street", "Block").
  Without stopword stripping, true pairs and unrelated same-DOB pairs overlapped heavily.
- **Threshold tuned by sweep** over 0.50-0.99, maximising F1 against the 40 known pairs, the same validation
  rigour as Stage 5.
- **`record_linkage_matches.national_id_no` renamed to `matched_record_id`** (`ALTER TABLE`), since the table now holds
  beneficiary-to-beneficiary matches; `match_method` = `'fuzzy_duplicate_identity'` disambiguates.
- Exact full-row duplicate artifacts (`is_duplicate_row`) are excluded from the comparison only (not deleted from the database).

### Bugs found and fixed
1. **Self-matches flooding the results.** Calling `recordlinkage.Index().index(df, df)` treats the inputs as two
   datasets, so rows matched to themselves (score 1.0). Fixed by using the single-argument deduplication form `index(df)`.
2. **Duplicate identities silently inheriting the original's bank account.** In the generator, the duplicate's
   disbursement rows were copied from the original's rows without overwriting `bank_account_number`, so every
   duplicate pair was also an unlogged `shared_bank_account` case. This inflated Stage 5's false positives
   (precision fell to 0.6369, with 171 FPs versus 61 explained hub owners). Fixed in the generator, with
   `patch_duplicate_identity_accounts.py` correcting the already-generated raw file, followed by a full re-run of Stages 2-6.

### Results (40 planted pairs)
| Dataset | Best threshold | Precision | Recall | F1 | Lowest true-pair score | Highest non-match score | Margin |
|---|---|---|---|---|---|---|---|
| First run | 0.57 | 1.0 | 1.0 | 1.0 | 0.585 | 0.569 | 0.016 |
| Current dataset | 0.55 | 1.0 | 1.0 | 1.0 | 0.576 | 0.540 | 0.036 |

On the current dataset: 40/40 true pairs found among candidates; true-pair scores min 0.576, median 0.961,
max 0.993; non-match scores median 0.323, 90th percentile 0.406, max 0.540; 44 pairs stored (score >= 0.50), 40 flagged.

### Limitations
Perfect scores validate detection of the specific planted perturbation (one character-level typo plus an abbreviation
swap), not generalisation to arbitrary real-world name variation. The margin between true pairs and the best
non-match is narrow.

---

## Stage 7 — Isolation Forest: Multivariate Behavioural Anomalies (2026-09-19)

**Status:** Complete, validated with limitations

### Goal
Detect genuinely registered beneficiaries whose payment behaviour is unusual across several features at once,
with no rule and no ground truth given to the model. Deliberately different from Stages 5-6, which target fake or
duplicated identities.

### What was built
- **`generate_anomalous_profiles.py`** (Stage 1 extension): plants 25 `multivariate_anomaly` profiles. It only edits
  `disbursements.csv`; identities are untouched. Candidates are Active beneficiaries with monthly income <= 8,000 who are not in any
  other planted pattern. Each gets: missing normal-window cycles filled, one extra payment in a cycle outside the
  simulated window (so exactly 7 payments, while legitimate beneficiaries can have at most 6), and up to 2 existing payments raised to
  the top of the normal amount pool (8,000 / 10,000). Answer key rows: `fraud_type='multivariate_anomaly'`, `beneficiary_id`.
- **`src/fraud_detection/isolation_forest.py`**: unsupervised detector writing to `fraud_signals`
  (`method='isolation_forest'`, `signal_type='multivariate_anomaly'`, score = anomaly score in (0,1), higher = more unusual).
- **`src/fraud_detection/evaluate_isolation_forest.py`**: scores the stored run against the answer key.

### Design decisions
- **Level of analysis:** one row per `beneficiary_id` (lowest `row_id` kept). `fraud_signals.entity_id` = that
  beneficiary's `row_id`, the same convention as Stages 5-6.
- **Features:** payment count, mean amount, total received, income (median-imputed if missing), amount-to-income
  (income floored at 1,000 to avoid dividing by zero).
- **Duplicate disbursement rows collapsed first** (same `disbursement_id`, from `inject_messiness.py`), otherwise a legitimate
  beneficiary with 6 payments plus one artifact copy looks like 7 payments.
- **Beneficiaries with no payments are not scored** (1,862 of 5,040); they have no payment behaviour to model.
- **Ground truth is used only in the evaluation script**, never by the detector.
- **Contamination 0.05** as the headline setting, 300 trees, seed 42. Contamination only moves the flagging cut-off, so the
  evaluation sweeps it without refitting. The default was not tuned against the answer key.
- **Layered mode (`--layered`)**: excludes beneficiaries already flagged by the verified Stage 5 `shared_bank_account` rule or
  Stage 6 fuzzy linkage (411 of them) from the model's training and scoring population for that run. Nothing is deleted from
  the database. Uses earlier detectors' output only.

### Evaluation design
Isolation Forest is unsupervised and finds "unusual", not one specific planted pattern, and other planted fraud is also unusual.
The evaluation therefore reports: precision/recall/F1 against `multivariate_anomaly`; a breakdown of what the flagged set
contains (planted anomaly / shared-account victim / duplicate identity / unlabeled); a contamination sweep; ROC-AUC and average
precision; and the population not already covered by Stages 5-6. Unlabeled flags are either novel findings or legitimate rare
behaviour and cannot be scored as either.

### Run 1: single population (3,178 scored beneficiaries, 159 flagged at 5%)
- Against the answer key: TP=3, FP=156, FN=22 (precision 0.0189, recall 0.12, F1 0.0326).
- Flagged set: 3 planted anomalies, 111 shared-account victims, 11 duplicate-identity beneficiaries, 34 unlabeled.
- Recall by flagging depth: 3/25 at 5%, 9/25 at 8%, 13/25 at 10%, 20/25 at 15%.
- Ranking quality: ROC-AUC 0.9003 (whole population), 0.9519 (planted versus ordinary only); average precision 0.1165 there,
  against 0.0079 for random guessing. As a sanity check, refitting with 10 seeds on the exported scores (using "exactly 7
  payments" as a stand-in label) gave ROC-AUC 0.941 +/- 0.007.
- Independent discovery: without any rules, 37% of shared-account victims (111/300) and 17.5% of scoreable
  duplicate-identity beneficiaries (11/63) were flagged.
- **Diagnosis:** shared-account victims are paid 9-12 times (their own plus hub payments), so they are more extreme than the planted
  7-payment profiles and use up the flagging budget. They also stretch the payment-count range the forest splits over, which
  makes a step from 6 to 7 payments harder to isolate.

### Run 2: layered (2,767 scored beneficiaries, 139 flagged at 5%)
- Against the answer key: TP=21, FP=118, FN=4 (precision 0.1511, recall 0.84, F1 0.2561).
- Recall by flagging depth: 13/25 at 3%, 21/25 at 5%, 23/25 at 8%, 25/25 at 10% (277 flagged, precision 0.09).
- ROC-AUC 0.9719, average precision 0.1847 (random guessing 0.0090).
- Top of the ranking: 1 planted anomaly in the top 10, 3 in the top 25, 6 in the top 50, 16 in the top 100. The planted profiles sit in
  the top ~5% band, not at the very top.

| | Run 1 (single population) | Run 2 (layered) |
|---|---|---|
| Planted found at 5% flagged | 3/25 (12%) | 21/25 (84%) |
| Planted found at 10% flagged | 13/25 (52%) | 25/25 (100%) |
| Precision at 5% | 0.019 | 0.151 |
| ROC-AUC (planted vs ordinary) | 0.952 | 0.972 |
| Average precision | 0.117 | 0.185 |

### Reference baseline (not a Stage 7 detector)
A single rule, `n_payments > 6`, flags exactly the 25 planted profiles: precision 1.0, recall 1.0. The planted pattern has a
deterministic single-feature separator. Isolation Forest is never told that feature or ceiling; the baseline shows what an analyst who already
knew where to look would achieve. Both are reported.

### Limitations
- The planted anomaly is separable by one threshold, so this experiment does not show Isolation Forest beating a rule.
- Precision stays low: 118 of 139 flagged at 5% (layered) are unlabeled.
- The layered change was chosen after seeing Run 1's flagged-set breakdown, a small evaluation feedback loop like Stage 6's threshold
  tuning. It was one structural change, with no parameter sweep.
- `inject_messiness.py`'s missing-value and outlier steps are not protected by `protected_ids` (only row duplication is), so a planted row could
  in principle be nulled during cleaning. Accepted, as in Stage 6.
- High scores on planted data do not prove real-world generalisation.

---

## Stage 8 — Comparative Evaluation of the Three Methods

**Status:** Not started

*(To be filled in: a head-to-head table of rule-based, fuzzy linkage and Isolation Forest against the same answer key;
whether and how to combine signals into a risk score, noting that `shared_bank_account` and `duplicate_cycle_payment` are not
independent evidence.)*

---

## Stage 9 — Orchestration

**Status:** Not started

*(To be filled in: Airflow DAG design, or fallback approach if Airflow setup was skipped, and why.)*

---

## Stage 10 — Dashboard

**Status:** Not started

*(To be filled in: dashboard pages built and key design choices.)*

---

## Open Questions / Follow-ups
- [ ] Confirm with instructor: full Airflow deployment vs. simplified scheduler acceptable?
- [ ] Confirm whether a secondary real public dataset should be blended in for validation.
- [ ] Decide which Isolation Forest configuration is the headline in the final report (single population or layered) and keep the other as the before/after.
- [ ] `is_duplicate_row` uses `duplicated(keep=False)`, so both copies of a messiness-duplicated beneficiary are flagged, and the fuzzy-linkage stage and the duplicate-CNIC rule then exclude both. Verify against the database and decide whether to keep one copy per `beneficiary_id`.
- [ ] Optionally protect planted rows from `inject_messiness.py`'s missing-value/outlier steps (known gap shared by Stages 6-7).