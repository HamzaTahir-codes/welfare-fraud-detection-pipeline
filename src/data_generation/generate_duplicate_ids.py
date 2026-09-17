"""
generate_duplicate_identities.py

Plants a third fraud pattern for the welfare fraud-detection pipeline:
"duplicate identity" ghost beneficiaries — the same real person registered
a second time under a new beneficiary_id, with name/address perturbed but
DOB unchanged, a fabricated (different) CNIC, a different bank account, and
its own disbursement history.

This is designed to be the ground-truth source for Stage 6 (fuzzy record
linkage via RapidFuzz/recordlinkage), the way generate_disbursements.py
already plants shared_bank_account and orphaned_disbursement fraud.

Run this AFTER generate_beneficiaries.py / generate_national_id.py /
generate_disbursements.py, and BEFORE inject_messiness.py — the messiness
step should treat every column this script writes as protected, since we
are deliberately injecting a controlled amount of noise here already.

>>> CHECK BEFORE RUNNING <<<
Column names below are inferred from project memory, not your actual
schema. Verify against your real CSVs (data/raw/*.csv) and adjust the
CONFIG section and column names in each function before running.
"""

import random
import re
import string
import uuid
from pathlib import Path

import pandas as pd
from faker import Faker

# ── CONFIG ────────────────────────────────────────────────────────────────
SEED = 42
N_DUPLICATE_PAIRS = 40

RAW_DIR = Path("data/raw")
BENEFICIARIES_PATH = RAW_DIR / "beneficiaries.csv"
NATIONAL_ID_PATH = RAW_DIR / "national_id.csv"
DISBURSEMENTS_PATH = RAW_DIR / "disbursements.csv"
GROUND_TRUTH_PATH = Path("data") / "ground_truth_fraud_ids.csv"

# Column names — VERIFY against your actual schema before running.
BENEFICIARY_ID_COL = "beneficiary_id"
FULL_NAME_COL = "full_name"
ADDRESS_COL = "address_line"
DOB_COL = "date_of_birth"
CNIC_COL = "cnic"
BANK_ACCOUNT_COL = "bank_account_number"
STATUS_COL = "status"

NATIONAL_ID_NO_COL = "national_id_no"
LINKED_BENEFICIARY_ID_COL = "linked_beneficiary_id"

DISBURSEMENT_ID_COL = "disbursement_id"
DISBURSEMENT_CYCLE_COL = "disbursement_cycle"
AMOUNT_COL = "amount_pkr"
PAYMENT_METHOD_COL = "payment_method"
DISBURSEMENT_DATE_COL = "disbursement_date"
DISBURSEMENT_STATUS_COL = "status"

MIN_DISBURSEMENTS_PER_DUPLICATE = 2
MAX_DISBURSEMENTS_PER_DUPLICATE = 6

random.seed(SEED)
fake = Faker()
Faker.seed(SEED)


# ── PERTURBATION HELPERS ────────────────────────────────────────────────────

def perturb_name(name: str) -> str:
    """Introduce a small realistic typo: one character swap, insertion,
    or deletion in a randomly chosen word of the name. Keeps the name
    recognizably similar (high RapidFuzz similarity) but not identical.
    """
    words = name.split()
    idx = random.randrange(len(words))
    word = list(words[idx])
    if len(word) < 3:
        words[idx] = word_str = "".join(word)
        return " ".join(words)

    op = random.choice(["swap", "insert", "delete"])
    pos = random.randrange(1, len(word) - 1)

    if op == "swap" and pos < len(word) - 1:
        word[pos], word[pos + 1] = word[pos + 1], word[pos]
    elif op == "insert":
        word.insert(pos, random.choice(string.ascii_lowercase))
    elif op == "delete":
        del word[pos]

    words[idx] = "".join(word)
    return " ".join(words)


def perturb_address(address: str) -> str:
    """Reword the address slightly: common abbreviation swap plus the
    same character-level typo used for names, applied to one word.
    """
    replacements = {
        "Street": "St",
        "St": "Street",
        "Road": "Rd",
        "Rd": "Road",
        "Avenue": "Ave",
        "Ave": "Avenue",
        "Colony": "Colny",
        "Block": "Blk",
    }
    words = address.split()
    for i, w in enumerate(words):
        stripped = w.strip(",")
        if stripped in replacements:
            words[i] = w.replace(stripped, replacements[stripped])
            return " ".join(words)

    # fall back to a character-level typo if no abbreviation found
    return perturb_name(address)


def fabricate_cnic() -> str:
    """Generate a syntactically valid but fabricated 13-digit CNIC,
    deliberately NOT derived from the original — this is the key
    difference from cnic_mismatch (typo) vs duplicate_identity (new ID).
    Adjust the format string if your CNICs aren't XXXXX-XXXXXXX-X.
    """
    return f"{random.randint(10000, 99999)}-{random.randint(1000000, 9999999)}-{random.randint(0, 9)}"


def fabricate_bank_account() -> str:
    """Generate a fabricated bank account number distinct from the
    original — adjust length/format to match your generator's convention.
    """
    return "".join(random.choices(string.digits, k=14))


# ── CORE GENERATION ──────────────────────────────────────────────────────

def load_data():
    beneficiaries = pd.read_csv(BENEFICIARIES_PATH)
    national_id = pd.read_csv(NATIONAL_ID_PATH)
    disbursements = pd.read_csv(DISBURSEMENTS_PATH)
    return beneficiaries, national_id, disbursements


def next_id_generator(existing_ids: pd.Series):
    """Yields new sequential IDs continuing after the current max, in the
    SAME format as the existing ones (prefix + zero-padding preserved).

    Detects prefix/width from an existing ID rather than hardcoding it
    (e.g. "BEN-007402" -> prefix "BEN-", width 6), so this keeps working
    if your ID scheme ever changes. Falls back to plain integers only if
    no ID in the column matches a "<prefix><digits>" pattern at all.
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


def plant_duplicate_identities(beneficiaries, national_id, disbursements, n_pairs=N_DUPLICATE_PAIRS):
    originals = beneficiaries.sample(n=n_pairs, random_state=SEED).copy()

    run_id = str(uuid.uuid4())  # one run_id per execution, matching the Stage 5 convention

    new_beneficiary_ids = next_id_generator(beneficiaries[BENEFICIARY_ID_COL])
    new_national_id_nos = set(national_id[NATIONAL_ID_NO_COL].astype(str))
    new_disbursement_ids = next_id_generator(disbursements[DISBURSEMENT_ID_COL])

    duplicate_beneficiary_rows = []
    duplicate_national_id_rows = []
    duplicate_disbursement_rows = []
    ground_truth_rows = []

    for _, orig in originals.iterrows():
        orig_id = orig[BENEFICIARY_ID_COL]
        dup_id = next(new_beneficiary_ids)

        dup_row = orig.copy()
        dup_row[BENEFICIARY_ID_COL] = dup_id
        dup_row[FULL_NAME_COL] = perturb_name(str(orig[FULL_NAME_COL]))
        dup_row[ADDRESS_COL] = perturb_address(str(orig[ADDRESS_COL]))
        # DOB deliberately left unchanged — same real person, real DOB doesn't vary
        dup_row[CNIC_COL] = fabricate_cnic()
        dup_row[BANK_ACCOUNT_COL] = fabricate_bank_account()
        dup_row[STATUS_COL] = "Active"
        duplicate_beneficiary_rows.append(dup_row)

        # matching national_id record for the duplicate, with its own fabricated CNIC
        new_nid = fabricate_cnic()
        while new_nid in new_national_id_nos:
            new_nid = fabricate_cnic()
        new_national_id_nos.add(new_nid)

        nid_row = {col: None for col in national_id.columns}
        nid_row[NATIONAL_ID_NO_COL] = new_nid
        nid_row[LINKED_BENEFICIARY_ID_COL] = dup_id
        duplicate_national_id_rows.append(nid_row)
        dup_row[CNIC_COL] = new_nid  # keep beneficiaries.cnic consistent with national_id_no

        # give the duplicate its own disbursement history
        n_disb = random.randint(MIN_DISBURSEMENTS_PER_DUPLICATE, MAX_DISBURSEMENTS_PER_DUPLICATE)
        template_disbs = disbursements[disbursements[BENEFICIARY_ID_COL] == orig_id]
        for _ in range(n_disb):
            disb_id = next(new_disbursement_ids)
            if len(template_disbs) > 0:
                template = template_disbs.sample(n=1, random_state=random.randint(0, 10_000)).iloc[0].copy()
            else:
                template = disbursements.sample(n=1, random_state=random.randint(0, 10_000)).iloc[0].copy()
            template[DISBURSEMENT_ID_COL] = disb_id
            template[BENEFICIARY_ID_COL] = dup_id
            template[BANK_ACCOUNT_COL] = dup_row[BANK_ACCOUNT_COL]  # use the duplicate's OWN fabricated
            # account, not the original's -- otherwise every planted duplicate-identity pair
            # accidentally also becomes an unlogged shared_bank_account case, inflating the
            # shared_bank_account rule's false-positive count against ground truth.
            duplicate_disbursement_rows.append(template)

        ground_truth_rows.append({
            "fraud_type": "duplicate_identity",
            "original_beneficiary_id": orig_id,
            "duplicate_beneficiary_id": dup_id,
            "run_id": run_id,
        })

    return (
        pd.DataFrame(duplicate_beneficiary_rows),
        pd.DataFrame(duplicate_national_id_rows),
        pd.DataFrame(duplicate_disbursement_rows),
        pd.DataFrame(ground_truth_rows),
    )


def append_and_save(beneficiaries, national_id, disbursements,
                     dup_beneficiaries, dup_national_id, dup_disbursements, ground_truth_new):
    beneficiaries_out = pd.concat([beneficiaries, dup_beneficiaries], ignore_index=True)
    national_id_out = pd.concat([national_id, dup_national_id], ignore_index=True)
    disbursements_out = pd.concat([disbursements, dup_disbursements], ignore_index=True)

    beneficiaries_out.to_csv(BENEFICIARIES_PATH, index=False)
    national_id_out.to_csv(NATIONAL_ID_PATH, index=False)
    disbursements_out.to_csv(DISBURSEMENTS_PATH, index=False)

    if GROUND_TRUTH_PATH.exists():
        existing_gt = pd.read_csv(GROUND_TRUTH_PATH)
        ground_truth_out = pd.concat([existing_gt, ground_truth_new], ignore_index=True)
    else:
        ground_truth_out = ground_truth_new
    ground_truth_out.to_csv(GROUND_TRUTH_PATH, index=False)

    # IDs to feed into inject_messiness.py's protected-rows mechanism —
    # both sides of every planted pair should be excluded from further
    # corruption, the same way existing planted fraud rows already are.
    protected_ids = set(ground_truth_new["original_beneficiary_id"]) | set(ground_truth_new["duplicate_beneficiary_id"])
    return protected_ids


def main():
    beneficiaries, national_id, disbursements = load_data()

    dup_beneficiaries, dup_national_id, dup_disbursements, ground_truth_new = plant_duplicate_identities(
        beneficiaries, national_id, disbursements, n_pairs=N_DUPLICATE_PAIRS
    )

    protected_ids = append_and_save(
        beneficiaries, national_id, disbursements,
        dup_beneficiaries, dup_national_id, dup_disbursements, ground_truth_new,
    )

    print(f"Planted {len(ground_truth_new)} duplicate-identity pairs.")
    print(f"New beneficiary rows: {len(dup_beneficiaries)}")
    print(f"New national_id rows: {len(dup_national_id)}")
    print(f"New disbursement rows: {len(dup_disbursements)}")
    print(f"Protected beneficiary_ids to exclude from inject_messiness.py: {len(protected_ids)}")
    # TODO: wire `protected_ids` into inject_messiness.py's existing
    # protected-rows set for planted fraud, so this controlled perturbation
    # isn't further scrambled by the general messiness step.


if __name__ == "__main__":
    main()