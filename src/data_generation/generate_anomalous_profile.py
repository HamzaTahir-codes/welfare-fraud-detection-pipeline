"""
generate_anomalous_profiles.py

Plants a fourth fraud/ground-truth pattern for the welfare fraud-detection
pipeline: "multivariate behavioral anomaly" -- beneficiaries whose IDENTITY
is completely genuine and untouched (no fake registration, no duplicated
person), but whose PAYMENT BEHAVIOR has been nudged, across several
features at once, into statistically unusual territory. This is the
ground-truth source for Stage 7 (Isolation Forest anomaly detection),
the way generate_disbursements.py already plants shared_bank_account /
orphaned_disbursement, and generate_duplicate_identities.py plants
duplicate_identity.

WHY THIS IS DESIGNED DIFFERENTLY FROM THE OTHER PLANTED PATTERNS:
Stage 5 (rule-based) and Stage 6 (fuzzy linkage) are both about a fake or
duplicated IDENTITY. Stage 7 is meant to demonstrate something a fixed
rule or an identity-matcher can't catch: a genuine, correctly-registered
beneficiary whose behavior is only MILDLY unusual on any single dimension,
but jointly unusual across a few dimensions at once. So this script never
touches beneficiaries.csv -- it only adds/adjusts rows in disbursements.csv
for beneficiaries who already exist.

Three features are nudged together, each one individually unremarkable:
  1. Average payment amount    -- raised using values ALREADY in the
                                   normal amount pool (8000 / 10000), never
                                   an out-of-range value.
  2. Payment count / frequency -- one extra payment is added in a cycle
                                   OUTSIDE the normal 6-month simulation
                                   window, pushing their count to 7 -- a
                                   ceiling no legitimately-generated
                                   beneficiary can reach (max is 6).
  3. Amount-to-income ratio    -- candidates are drawn only from
                                   beneficiaries already in the LOW income
                                   tiers, so an elevated payment average
                                   against a low, untouched, genuine income
                                   creates this ratio anomaly for free.

Run this AFTER generate_duplicate_identities.py and BEFORE
inject_messiness.py, matching the established ordering for planted
ground-truth patterns.

>>> CHECK BEFORE RUNNING <<<
Column names and value pools below are inferred from project memory, not
your actual schema/generator constants. Verify against generate_disbursements.py
(PAYMENT_METHODS, STATUS, amount choices) and your real CSVs before running.

KNOWN GAP (shared with generate_duplicate_identities.py, not fixed here):
inject_messiness.py's missing-value and outlier-forcing steps in
mess_disbursements() apply to the WHOLE disbursements table -- only
row-duplication currently respects protected_ids. There is a small
(~2%, this script's default outlier rate) chance a planted row here gets
hit by random outlier-forcing and nulled out during Stage 3 cleaning.
Left as an accepted risk, consistent with how this was left for Stage 6;
fix inject_messiness.py directly if you want both gaps closed.
"""

import random
import re
import string
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────
SEED = 42
N_ANOMALOUS_PROFILES = 25

# Candidates must already be in a low income tier so an elevated payment
# average creates a visible amount-vs-income mismatch without us ever
# touching their stated income.
LOW_INCOME_THRESHOLD = 8000

# Amounts used to fill/bump this beneficiary's payments -- deliberately the
# SAME values already present in generate_disbursements.py's normal pool
# (biased toward the top of it), so no single payment is out-of-range.
ELEVATED_AMOUNT_CHOICES = [8000, 10000, 10000]

RAW_DIR = Path("data/raw")
BENEFICIARIES_PATH = RAW_DIR / "beneficiaries.csv"
DISBURSEMENTS_PATH = RAW_DIR / "disbursements.csv"
GROUND_TRUTH_PATH = Path("data") / "ground_truth_fraud_ids.csv"

# Column names -- VERIFY against your actual schema before running.
BENEFICIARY_ID_COL = "beneficiary_id"
STATUS_COL = "status"
INCOME_COL = "monthly_income_pkr"
BANK_ACCOUNT_COL = "bank_account_number"

DISBURSEMENT_ID_COL = "disbursement_id"
DISBURSEMENT_CYCLE_COL = "disbursement_cycle"
AMOUNT_COL = "amount_pkr"
DATE_COL = "disbursement_date"
PAYMENT_METHOD_COL = "payment_method"
BRANCH_COL = "branch_or_agent_code"
DISB_STATUS_COL = "status"

# Mirrors generate_disbursements.py's constants -- ADJUST if yours differ.
PAYMENT_METHODS = ["Bank Transfer", "Mobile Wallet", "Cash Pickup"]
DISB_STATUS_CHOICES = ["Success", "Success", "Success", "Success", "Pending", "Reversed"]

random.seed(SEED)


# ── HELPERS ──────────────────────────────────────────────────────────────

def next_id_generator(existing_ids: pd.Series):
    """Yields new sequential IDs continuing after the current max, in the
    SAME format as the existing ones. Identical logic to the helper in
    generate_duplicate_identities.py -- kept local here so this script has
    no import dependency on that one.
    """
    ids = existing_ids.astype(str)
    match = re.match(r"^(\D*)(\d+)$", ids.iloc[0])
    if match:
        prefix, width = match.group(1), len(match.group(2))
    else:
        prefix, width = "", 0

    numeric_parts = ids.str.extract(r"(\d+)$")[0].astype(int)
    n = int(numeric_parts.max())
    while True:
        n += 1
        yield f"{prefix}{str(n).zfill(width)}"


def shift_cycle(cycle: str, months_back: int) -> str:
    """Shifts a 'YYYY-MM' cycle string back by months_back months."""
    year, month = map(int, cycle.split("-"))
    total = year * 12 + (month - 1) - months_back
    new_year, new_month = divmod(total, 12)
    return f"{new_year:04d}-{new_month + 1:02d}"


def random_date_in_cycle(cycle: str) -> str:
    year, month = map(int, cycle.split("-"))
    day = random.randint(1, 28)
    return datetime(year, month, day).strftime("%Y-%m-%d")


def make_disbursement_row(disb_id, beneficiary_id, bank_account, cycle) -> dict:
    return {
        DISBURSEMENT_ID_COL: disb_id,
        BENEFICIARY_ID_COL: beneficiary_id,
        BANK_ACCOUNT_COL: bank_account,
        AMOUNT_COL: random.choice(ELEVATED_AMOUNT_CHOICES),
        DISBURSEMENT_CYCLE_COL: cycle,
        DATE_COL: random_date_in_cycle(cycle),
        PAYMENT_METHOD_COL: random.choice(PAYMENT_METHODS),
        BRANCH_COL: f"BR-{random.randint(100, 299)}",
        DISB_STATUS_COL: random.choice(DISB_STATUS_CHOICES),
    }


# ── DATA LOADING ─────────────────────────────────────────────────────────

def load_data():
    beneficiaries = pd.read_csv(BENEFICIARIES_PATH)
    disbursements = pd.read_csv(DISBURSEMENTS_PATH)
    return beneficiaries, disbursements


def load_already_implicated_ids(gt_path: Path) -> set:
    """Every beneficiary_id already involved in ANY previously planted
    fraud pattern (shared_account, orphan_disbursement, duplicate_identity)
    -- excluded from candidacy here so this pattern stays a clean, separate
    signal rather than overlapping with an existing one.
    """
    if not gt_path.exists():
        return set()
    gt = pd.read_csv(gt_path)
    implicated = set()
    for col in ("beneficiary_id", "original_beneficiary_id", "duplicate_beneficiary_id"):
        if col in gt.columns:
            implicated |= set(gt[col].dropna())
    return implicated


# ── CORE GENERATION ──────────────────────────────────────────────────────

def select_candidates(beneficiaries: pd.DataFrame, excluded_ids: set, n: int) -> pd.DataFrame:
    eligible = beneficiaries[
        (beneficiaries[STATUS_COL] == "Active")
        & (beneficiaries[INCOME_COL] <= LOW_INCOME_THRESHOLD)
        & (~beneficiaries[BENEFICIARY_ID_COL].isin(excluded_ids))
    ]
    if len(eligible) < n:
        raise ValueError(
            f"Only {len(eligible)} eligible low-income active beneficiaries available, "
            f"need {n}. Lower N_ANOMALOUS_PROFILES or raise LOW_INCOME_THRESHOLD."
        )
    return eligible.sample(n=n, random_state=SEED)


def plant_anomalous_profiles(beneficiaries: pd.DataFrame, disbursements: pd.DataFrame, n_profiles: int):
    run_id = str(uuid.uuid4())
    already_implicated = load_already_implicated_ids(GROUND_TRUTH_PATH)
    candidates = select_candidates(beneficiaries, already_implicated, n_profiles)

    all_cycles = sorted(disbursements[DISBURSEMENT_CYCLE_COL].dropna().unique())
    earliest_cycle = all_cycles[0]
    bonus_cycle = shift_cycle(earliest_cycle, months_back=1)  # outside the normal window

    id_gen = next_id_generator(disbursements[DISBURSEMENT_ID_COL])
    new_rows = []
    ground_truth_rows = []
    amount_bump_index = []  # indices into `disbursements` whose amount gets nudged in place

    for _, ben in candidates.iterrows():
        bid = ben[BENEFICIARY_ID_COL]
        bank_account = ben[BANK_ACCOUNT_COL]

        existing = disbursements[disbursements[BENEFICIARY_ID_COL] == bid]
        present_cycles = set(existing[DISBURSEMENT_CYCLE_COL].dropna())

        # 1) Fill any of the normal 6 cycles this beneficiary happened to miss
        #    -- mild, secondary attendance-consistency bump.
        for cycle in all_cycles:
            if cycle not in present_cycles:
                new_rows.append(make_disbursement_row(next(id_gen), bid, bank_account, cycle))

        # 2) One extra payment in a cycle OUTSIDE the simulated window --
        #    this is the deterministic frequency signal: pushes their total
        #    count to a ceiling no legitimately-generated beneficiary can reach.
        new_rows.append(make_disbursement_row(next(id_gen), bid, bank_account, bonus_cycle))

        # 3) Nudge up to 2 of this beneficiary's EXISTING payments to the
        #    top of the normal amount pool, in place -- raises their average
        #    further without adding any new row or any out-of-range value.
        if len(existing.index) > 0:
            bump_idx = random.sample(list(existing.index), k=min(2, len(existing.index)))
            amount_bump_index.extend(bump_idx)

        ground_truth_rows.append({
            "fraud_type": "multivariate_anomaly",
            "beneficiary_id": bid,
            "run_id": run_id,
        })

    disbursements = disbursements.copy()
    if amount_bump_index:
        disbursements.loc[amount_bump_index, AMOUNT_COL] = [
            random.choice(ELEVATED_AMOUNT_CHOICES) for _ in amount_bump_index
        ]

    disbursements_out = pd.concat([disbursements, pd.DataFrame(new_rows)], ignore_index=True)
    return disbursements_out, pd.DataFrame(ground_truth_rows)


def append_and_save(disbursements_out: pd.DataFrame, ground_truth_new: pd.DataFrame):
    disbursements_out.to_csv(DISBURSEMENTS_PATH, index=False)

    if GROUND_TRUTH_PATH.exists():
        existing_gt = pd.read_csv(GROUND_TRUTH_PATH)
        ground_truth_out = pd.concat([existing_gt, ground_truth_new], ignore_index=True)
    else:
        ground_truth_out = ground_truth_new
    ground_truth_out.to_csv(GROUND_TRUTH_PATH, index=False)

    # beneficiary_id is populated directly in ground_truth_fraud_ids.csv for
    # this pattern (unlike duplicate_identity's original/duplicate columns),
    # so inject_messiness.py's existing load_ground_truth_keys() already
    # picks these up automatically for row-duplication protection -- no
    # further wiring needed for that specific protection.
    return set(ground_truth_new["beneficiary_id"])


def main():
    beneficiaries, disbursements = load_data()

    disbursements_out, ground_truth_new = plant_anomalous_profiles(
        beneficiaries, disbursements, n_profiles=N_ANOMALOUS_PROFILES
    )

    protected_ids = append_and_save(disbursements_out, ground_truth_new)

    print(f"Planted {len(ground_truth_new)} multivariate-anomaly beneficiary profiles.")
    print(f"New/adjusted disbursement rows added: {len(disbursements_out) - len(disbursements)}")
    print(f"Protected beneficiary_ids (already covered by inject_messiness.py's "
          f"row-duplication protection): {len(protected_ids)}")
    print("\nNOTE: inject_messiness.py's missing-value/outlier-forcing steps still apply "
          "to the whole disbursements table regardless of protected_ids (same known gap "
          "as generate_duplicate_identities.py). Small chance a planted row gets nulled "
          "by random outlier-forcing during Stage 3 cleaning -- see module docstring.")


if __name__ == "__main__":
    main()