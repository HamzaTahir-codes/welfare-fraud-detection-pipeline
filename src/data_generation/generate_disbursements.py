"""
generate_disbursements.py

Generates a realistic, synthetic Bank Disbursement Log
(Stage 1 - Simulated Source Systems, Source #3 of 3).

This is where the clearest fraud signals get planted:

1. SHARED ACCOUNT FRAUD (the classic "ghost beneficiary" pattern):
   A small set of bank accounts are deliberately reused across MULTIPLE
   different beneficiary_ids -- simulating a fraudster who registered
   several fake/duplicate identities but funnels all the payments into
   one real account they control.

2. ORPHAN DISBURSEMENTS:
   A small number of disbursement records reference a beneficiary_id
   that does NOT exist in beneficiaries.csv at all -- simulating a
   payment made against a completely fabricated registration.

Every planted fraud case is written to a separate ground-truth file
(`data/ground_truth_fraud_ids.csv`) that your ETL/cleaning code should
NEVER read from -- it exists only so Stage 6 (fraud-detection
experimentation) can later score precision/recall honestly.

Usage:
    python src/data_generation/generate_disbursements.py
    python src/data_generation/generate_disbursements.py --cycles 6 --fraud-account-rate 0.03
"""

import argparse
import random
import string
from datetime import datetime
from pathlib import Path

import pandas as pd
from faker import Faker


PAYMENT_METHODS = ["Bank Transfer", "Mobile Wallet", "Cash Pickup"]
STATUS = ["Success", "Success", "Success", "Success", "Pending", "Reversed"]


def month_cycles(n_cycles: int) -> list:
    """Returns a list of the last n_cycles year-month strings, e.g. ['2025-06', '2025-07', ...]."""
    cycles = []
    year, month = 2025, 7  # anchor point; adjust as needed
    for _ in range(n_cycles):
        cycles.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return list(reversed(cycles))


def random_date_in_cycle(cycle: str) -> str:
    year, month = map(int, cycle.split("-"))
    day = random.randint(1, 28)
    return datetime(year, month, day).strftime("%Y-%m-%d")


def generate_normal_disbursements(beneficiaries: pd.DataFrame, cycles: list) -> list:
    """One disbursement per active beneficiary per cycle, using their own registered account."""
    records = []
    disb_counter = 1

    active_beneficiaries = beneficiaries[beneficiaries["status"] == "Active"]

    for _, row in active_beneficiaries.iterrows():
        for cycle in cycles:
            # small chance a payment is simply skipped that cycle (realistic gaps)
            if random.random() < 0.08:
                continue

            record = {
                "disbursement_id": f"DSB-{disb_counter:07d}",
                "beneficiary_id": row["beneficiary_id"],
                "bank_account_number": row["bank_account_number"],
                "amount_pkr": random.choice([5000, 6000, 7500, 8000, 10000]),
                "disbursement_cycle": cycle,
                "disbursement_date": random_date_in_cycle(cycle),
                "payment_method": random.choice(PAYMENT_METHODS),
                "branch_or_agent_code": f"BR-{random.randint(100,299)}",
                "status": random.choice(STATUS),
            }
            records.append(record)
            disb_counter += 1

    return records, disb_counter


def plant_shared_account_fraud(records: list, beneficiaries: pd.DataFrame, fraud_account_rate: float, start_counter: int, cycles: list):
    """
    Picks a small set of 'hub' accounts and reassigns a handful of OTHER
    beneficiaries' disbursements to funnel into those same accounts --
    simulating one person controlling several registered identities.
    Returns the new records plus a ground-truth log of what was planted.
    """
    ground_truth = []
    disb_counter = start_counter

    n_hub_accounts = max(1, int(len(beneficiaries) * fraud_account_rate))
    hub_accounts = beneficiaries.sample(n=n_hub_accounts)["bank_account_number"].tolist()

    # For each hub account, pick 2-4 OTHER beneficiaries to funnel into it
    remaining_pool = beneficiaries.sample(frac=1).to_dict("records")
    pool_idx = 0

    for hub_account in hub_accounts:
        n_victims = random.randint(2, 4)
        for _ in range(n_victims):
            if pool_idx >= len(remaining_pool):
                break
            victim = remaining_pool[pool_idx]
            pool_idx += 1

            for cycle in cycles:
                if random.random() < 0.1:
                    continue
                record = {
                    "disbursement_id": f"DSB-{disb_counter:07d}",
                    "beneficiary_id": victim["beneficiary_id"],
                    "bank_account_number": hub_account,  # <-- the planted overlap
                    "amount_pkr": random.choice([5000, 6000, 7500, 8000, 10000]),
                    "disbursement_cycle": cycle,
                    "disbursement_date": random_date_in_cycle(cycle),
                    "payment_method": random.choice(PAYMENT_METHODS),
                    "branch_or_agent_code": f"BR-{random.randint(100,299)}",
                    "status": random.choice(STATUS),
                }
                records.append(record)
                disb_counter += 1

            ground_truth.append({
                "fraud_type": "shared_account",
                "bank_account_number": hub_account,
                "beneficiary_id": victim["beneficiary_id"],
            })

    return records, ground_truth, disb_counter


def plant_orphan_disbursements(records: list, n_orphans: int, start_counter: int, cycles: list, fake: Faker):
    """Creates disbursement records for beneficiary_ids that don't exist anywhere in the registry."""
    ground_truth = []
    disb_counter = start_counter

    for i in range(n_orphans):
        fake_beneficiary_id = f"BEN-{900000 + i:06d}"  # deliberately outside the real ID range
        fake_account = "PK" + "".join(random.choices(string.digits, k=16))
        cycle = random.choice(cycles)

        record = {
            "disbursement_id": f"DSB-{disb_counter:07d}",
            "beneficiary_id": fake_beneficiary_id,
            "bank_account_number": fake_account,
            "amount_pkr": random.choice([5000, 6000, 7500, 8000, 10000]),
            "disbursement_cycle": cycle,
            "disbursement_date": random_date_in_cycle(cycle),
            "payment_method": random.choice(PAYMENT_METHODS),
            "branch_or_agent_code": f"BR-{random.randint(100,299)}",
            "status": "Success",
        }
        records.append(record)
        ground_truth.append({
            "fraud_type": "orphan_disbursement",
            "bank_account_number": fake_account,
            "beneficiary_id": fake_beneficiary_id,
        })
        disb_counter += 1

    return records, ground_truth, disb_counter


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic bank disbursement log with planted fraud.")
    parser.add_argument("--beneficiaries-file", type=str, default="data/raw/beneficiaries.csv")
    parser.add_argument("--cycles", type=int, default=6, help="Number of monthly disbursement cycles to simulate.")
    parser.add_argument("--fraud-account-rate", type=float, default=0.02,
                         help="Fraction of beneficiaries whose accounts become 'hub' fraud accounts.")
    parser.add_argument("--orphan-count", type=int, default=25,
                         help="Number of disbursements paid to completely fabricated beneficiary IDs.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default="data/raw/disbursements.csv")
    parser.add_argument("--ground-truth-output", type=str, default="data/ground_truth_fraud_ids.csv")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        Faker.seed(args.seed)

    fake = Faker()

    beneficiaries_path = Path(args.beneficiaries_file)
    if not beneficiaries_path.exists():
        raise FileNotFoundError(f"Could not find {beneficiaries_path}. Run generate_beneficiaries.py first.")

    beneficiaries = pd.read_csv(beneficiaries_path)
    print(f"Loaded {len(beneficiaries)} beneficiaries from {beneficiaries_path}")

    cycles = month_cycles(args.cycles)
    print(f"Simulating disbursement cycles: {cycles}")

    records, counter = generate_normal_disbursements(beneficiaries, cycles)
    print(f"Generated {len(records)} normal disbursement records.")

    records, gt_shared, counter = plant_shared_account_fraud(
        records, beneficiaries, args.fraud_account_rate, counter, cycles
    )
    print(f"Planted shared-account fraud: {len(gt_shared)} beneficiary-account overlaps.")

    records, gt_orphan, counter = plant_orphan_disbursements(
        records, args.orphan_count, counter, cycles, fake
    )
    print(f"Planted {len(gt_orphan)} orphan disbursements (fabricated beneficiary IDs).")

    df = pd.DataFrame(records).sample(frac=1).reset_index(drop=True)  # shuffle
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    ground_truth_df = pd.DataFrame(gt_shared + gt_orphan)
    gt_path = Path(args.ground_truth_output)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_df.to_csv(gt_path, index=False)

    print(f"\nDone. Saved {len(df)} disbursement records to: {output_path.resolve()}")
    print(f"Saved {len(ground_truth_df)} planted fraud entries to: {gt_path.resolve()}")
    print("\nSample normal record:")
    print(df.iloc[0].to_string())


if __name__ == "__main__":
    main()