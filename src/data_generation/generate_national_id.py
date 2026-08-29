"""
generate_national_id.py

Generates a realistic, synthetic National ID / Address database
(Stage 1 - Simulated Source Systems, Source #2 of 3).

Design idea: a real national ID database is NOT the same system as the
welfare beneficiary registry -- it's maintained by a different government
department, so records that describe the SAME person often don't match
perfectly (typos, reformatted addresses, minor date differences). This
script deliberately recreates that mismatch, which is exactly the kind of
messiness your ETL/cleaning stage (Stage 3) needs to handle.

It does two things:
1. For most beneficiaries, creates a corresponding national ID record --
   but with realistic data-entry noise (typo'd names, reformatted
   addresses, occasional DOB drift).
2. Deliberately leaves a small percentage of beneficiaries WITHOUT a
   matching national ID record -- simulating identities that don't trace
   back to a real citizen (a classic ghost-beneficiary red flag you'll
   pick up on later in the fraud-detection stage).
3. Adds extra "citizen-only" records not linked to any beneficiary, since
   a real national ID database covers the whole population, not just
   welfare recipients.

Usage:
    python src/data_generation/generate_national_id.py
    python src/data_generation/generate_national_id.py --match-rate 0.88 --extra-citizens 2000
"""

import argparse
import random
import string
from pathlib import Path

import pandas as pd
from faker import Faker


ID_STATUS = ["Active", "Active", "Active", "Expired"]


def maybe_typo_name(name: str, typo_chance: float = 0.15) -> str:
    """Occasionally introduces a small human-error-style typo into a name."""
    if random.random() > typo_chance or len(name) < 4:
        return name

    name_chars = list(name)
    typo_type = random.choice(["swap", "drop", "case"])
    idx = random.randint(0, len(name_chars) - 2)

    if typo_type == "swap":
        name_chars[idx], name_chars[idx + 1] = name_chars[idx + 1], name_chars[idx]
    elif typo_type == "drop":
        del name_chars[idx]
    elif typo_type == "case":
        name_chars[idx] = name_chars[idx].upper() if name_chars[idx].islower() else name_chars[idx].lower()

    return "".join(name_chars)


def maybe_mutate_cnic(cnic: str, mutate_chance: float = 0.08) -> str:
    """Occasionally mutates one digit of a CNIC to simulate a data-entry error."""
    if random.random() > mutate_chance:
        return cnic

    digits = list(cnic)
    idx = random.choice([i for i, c in enumerate(cnic) if c.isdigit()])
    digits[idx] = random.choice(string.digits)
    return "".join(digits)


def reformat_address(address: str, city: str, province: str) -> str:
    """
    Reformats an address the way a *different* government system might
    store it -- e.g. abbreviated, reordered, or missing the street type.
    """
    style = random.choice(["full", "abbreviated", "reordered"])
    if style == "full":
        return f"{address}, {city}, {province}"
    elif style == "abbreviated":
        short_addr = address.replace("Street", "St").replace("Road", "Rd").replace("Avenue", "Ave")
        return f"{short_addr}, {city}"
    else:  # reordered
        return f"{province}, {city}, {address}"


def drift_dob(dob_str: str, drift_chance: float = 0.05) -> str:
    """Occasionally shifts the date of birth by a day or two (transcription error)."""
    from datetime import datetime, timedelta

    if random.random() > drift_chance:
        return dob_str

    dob = datetime.strptime(dob_str, "%Y-%m-%d")
    dob += timedelta(days=random.choice([-2, -1, 1, 2]))
    return dob.strftime("%Y-%m-%d")


def generate_extra_citizen(fake: Faker, next_id: int) -> dict:
    """Generates a citizen record with no link to any beneficiary (population noise)."""
    gender = random.choice(["Male", "Female"])
    full_name = fake.name_male() if gender == "Male" else fake.name_female()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=90)
    cnic = f"{random.randint(10000,99999)}-{random.randint(1000000,9999999)}-{random.randint(0,9)}"

    return {
        "national_id_no": cnic,
        "full_name": full_name,
        "gender": gender,
        "date_of_birth": dob.strftime("%Y-%m-%d"),
        "address": reformat_address(fake.street_address(), fake.city(), "Punjab"),
        "id_issue_date": fake.date_between(start_date="-15y", end_date="-1y").strftime("%Y-%m-%d"),
        "id_status": random.choice(ID_STATUS),
        "linked_beneficiary_id": None,  # not a welfare recipient
    }


def generate_national_id_records(beneficiaries: pd.DataFrame, match_rate: float, extra_citizens: int, fake: Faker) -> pd.DataFrame:
    records = []

    # Decide which beneficiaries get a matching national ID record
    sample_size = int(len(beneficiaries) * match_rate)
    matched_beneficiaries = beneficiaries.sample(n=sample_size, random_state=None)

    for _, row in matched_beneficiaries.iterrows():
        record = {
            "national_id_no": maybe_mutate_cnic(row["cnic"]),
            "full_name": maybe_typo_name(row["full_name"]),
            "gender": row["gender"],
            "date_of_birth": drift_dob(row["date_of_birth"]),
            "address": reformat_address(row["address_line"], row["city"], row["province"]),
            "id_issue_date": fake.date_between(start_date="-15y", end_date="-1y").strftime("%Y-%m-%d"),
            "id_status": random.choice(ID_STATUS),
            "linked_beneficiary_id": row["beneficiary_id"],
        }
        records.append(record)

    # Add extra citizen-only records (population beyond welfare recipients)
    for i in range(extra_citizens):
        records.append(generate_extra_citizen(fake, i))

    df = pd.DataFrame(records)
    return df.sample(frac=1).reset_index(drop=True)  # shuffle rows


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic national ID / address database.")
    parser.add_argument(
        "--beneficiaries-file",
        type=str,
        default="data/raw/beneficiaries.csv",
        help="Path to the beneficiaries CSV generated by generate_beneficiaries.py",
    )
    parser.add_argument(
        "--match-rate",
        type=float,
        default=0.90,
        help="Fraction of beneficiaries who WILL have a matching national ID record (rest simulate missing/ghost identities).",
    )
    parser.add_argument(
        "--extra-citizens",
        type=int,
        default=1500,
        help="Number of extra citizen-only records not linked to any beneficiary.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/national_id_records.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        Faker.seed(args.seed)

    fake = Faker()

    beneficiaries_path = Path(args.beneficiaries_file)
    if not beneficiaries_path.exists():
        raise FileNotFoundError(
            f"Could not find {beneficiaries_path}. Run generate_beneficiaries.py first."
        )

    beneficiaries = pd.read_csv(beneficiaries_path)
    print(f"Loaded {len(beneficiaries)} beneficiaries from {beneficiaries_path}")

    print(
        f"Generating national ID records: {args.match_rate:.0%} match rate, "
        f"{args.extra_citizens} extra citizen-only records..."
    )
    df = generate_national_id_records(beneficiaries, args.match_rate, args.extra_citizens, fake)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    unmatched_count = len(beneficiaries) - int(len(beneficiaries) * args.match_rate)
    print(f"Done. Saved {len(df)} records to: {output_path.resolve()}")
    print(f"  -> {int(len(beneficiaries) * args.match_rate)} records linked to real beneficiaries")
    print(f"  -> {args.extra_citizens} extra citizen-only records")
    print(f"  -> {unmatched_count} beneficiaries were LEFT WITHOUT a national ID match (potential ghost identities)")
    print("\nSample record:")
    print(df.iloc[0].to_string())


if __name__ == "__main__":
    main()