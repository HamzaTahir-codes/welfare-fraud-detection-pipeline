"""
generate_beneficiaries.py

Generates a realistic, synthetic Beneficiary Registry dataset for the
Welfare Fraud Detection pipeline (Stage 1 - Simulated Source Systems).

This represents Source #1 of 3: the beneficiary registry a welfare
program would maintain when someone enrolls.

Usage:
    python src/data_generation/generate_beneficiaries.py --count 5000
    python src/data_generation/generate_beneficiaries.py --count 5000 --seed 42
"""

import argparse
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker


# ---------------------------------------------------------------------------
# Reference lists — kept realistic for a Pakistani welfare-program context.
# Swap these out if you want to target a different country/region.
# ---------------------------------------------------------------------------

PROVINCES_CITIES = {
    "Punjab": ["Lahore", "Multan", "Faisalabad", "Rawalpindi", "Gujranwala", "Sialkot", "Bahawalpur"],
    "Sindh": ["Karachi", "Hyderabad", "Sukkur", "Larkana"],
    "Khyber Pakhtunkhwa": ["Peshawar", "Mardan", "Abbottabad", "Swat"],
    "Balochistan": ["Quetta", "Gwadar", "Sibi"],
}

WELFARE_PROGRAMS = [
    "Cash Transfer Program",
    "Disability Support Program",
    "Widow & Orphan Support Program",
    "Flood Relief Assistance",
    "Elderly Pension Program",
]

MARITAL_STATUS = ["Single", "Married", "Widowed", "Divorced"]
REGISTRATION_CHANNEL = ["In-Person Center", "Mobile Registration Van", "Online Portal", "NGO Referral"]
STATUS = ["Active", "Active", "Active", "Inactive", "Suspended"]  # weighted toward Active


def generate_cnic(fake: Faker) -> str:
    """Generates a Pakistani-style CNIC number: 5 digits - 7 digits - 1 digit."""
    part1 = "".join(random.choices(string.digits, k=5))
    part2 = "".join(random.choices(string.digits, k=7))
    part3 = random.choice(string.digits)
    return f"{part1}-{part2}-{part3}"


def generate_phone(fake: Faker) -> str:
    """Generates a Pakistani-style mobile number: 03XX-XXXXXXX."""
    prefix = random.choice(["300", "301", "302", "303", "310", "320", "333", "345"])
    number = "".join(random.choices(string.digits, k=7))
    return f"0{prefix}-{number}"


def generate_bank_account(fake: Faker) -> str:
    """Generates a mock bank account / IBAN-style reference used for disbursement matching."""
    return "PK" + "".join(random.choices(string.digits, k=16))


def random_registration_date() -> str:
    """Random registration date within the last 6 years."""
    start = datetime(2019, 1, 1)
    end = datetime(2025, 12, 31)
    delta_days = (end - start).days
    reg_date = start + timedelta(days=random.randint(0, delta_days))
    return reg_date.strftime("%Y-%m-%d")


def generate_beneficiaries(n: int, fake: Faker) -> pd.DataFrame:
    records = []

    for i in range(1, n + 1):
        gender = random.choice(["Male", "Female"])
        full_name = fake.name_male() if gender == "Male" else fake.name_female()
        father_or_husband_name = fake.name_male()  # commonly required on welfare forms
        province = random.choice(list(PROVINCES_CITIES.keys()))
        city = random.choice(PROVINCES_CITIES[province])
        dob = fake.date_of_birth(minimum_age=18, maximum_age=85)
        household_size = random.randint(1, 9)
        monthly_income = random.choice([0, 5000, 8000, 12000, 15000, 20000, 25000])

        record = {
            "beneficiary_id": f"BEN-{i:06d}",
            "full_name": full_name,
            "father_or_husband_name": father_or_husband_name,
            "gender": gender,
            "date_of_birth": dob.strftime("%Y-%m-%d"),
            "cnic": generate_cnic(fake),
            "phone_number": generate_phone(fake),
            "marital_status": random.choice(MARITAL_STATUS),
            "household_size": household_size,
            "monthly_income_pkr": monthly_income,
            "address_line": fake.street_address(),
            "city": city,
            "province": province,
            "bank_account_number": generate_bank_account(fake),
            "welfare_program": random.choice(WELFARE_PROGRAMS),
            "registration_date": random_registration_date(),
            "registration_channel": random.choice(REGISTRATION_CHANNEL),
            "status": random.choice(STATUS),
        }
        records.append(record)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic beneficiary registry data.")
    parser.add_argument("--count", type=int, default=5000, help="Number of beneficiary records to generate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/beneficiaries.csv",
        help="Output CSV path (relative to project root).",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        Faker.seed(args.seed)

    fake = Faker()  # default locale; swap to Faker("en_US") etc. if you prefer

    print(f"Generating {args.count} synthetic beneficiary records...")
    df = generate_beneficiaries(args.count, fake)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Done. Saved {len(df)} records to: {output_path.resolve()}")
    print("\nSample record:")
    print(df.iloc[0].to_string())


if __name__ == "__main__":
    main()