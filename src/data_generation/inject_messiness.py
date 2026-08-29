"""
inject_messiness.py

Takes the already-generated raw CSVs (beneficiaries, national_id_records,
disbursements) and deliberately injects realistic data-quality problems --
missing values, inconsistent formatting, negative/outlier numbers,
inconsistent category labels, duplicate rows, and mixed date formats.

This simulates what real government/bank data actually looks like after
years of manual entry across different departments -- and gives you real
problems to solve in Stage 3 (clean_transform.py).

IMPORTANT: this script never touches identifier/key columns
(beneficiary_id, cnic, national_id_no, bank_account_number,
linked_beneficiary_id, disbursement_id) -- corrupting those would break
the record-linkage and fraud-detection logic that depends on them later.
It also never drops or duplicates rows that appear in your planted
ground-truth fraud list, so your fraud signals stay intact.

Run this AFTER the 3 generate_*.py scripts, and BEFORE building
extract_stage.py / clean_transform.py.

Usage:
    python src/data_generation/inject_messiness.py
    python src/data_generation/inject_messiness.py --missing-rate 0.08 --outlier-rate 0.03
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd


PROTECTED_COLUMNS = {
    "beneficiary_id", "cnic", "national_id_no",
    "bank_account_number", "linked_beneficiary_id", "disbursement_id",
}


def load_ground_truth_keys(gt_path: Path) -> set:
    """Loads beneficiary_ids involved in planted fraud, so we never drop/duplicate those rows."""
    if not gt_path.exists():
        return set()
    gt = pd.read_csv(gt_path)
    return set(gt["beneficiary_id"].dropna())


# ---------------------------------------------------------------------------
# Generic messiness helpers
# ---------------------------------------------------------------------------

def randomly_blank(series: pd.Series, rate: float) -> pd.Series:
    """Sets a random fraction of values to NaN."""
    series = series.copy()
    mask = np.random.random(len(series)) < rate
    series[mask] = np.nan
    return series


def randomly_whitespace_and_case(series: pd.Series, rate: float) -> pd.Series:
    """Adds stray whitespace and inconsistent casing to text values."""
    def mess(val):
        if pd.isna(val) or random.random() > rate:
            return val
        val = str(val)
        style = random.choice(["upper", "lower", "leading_space", "trailing_space", "double_space"])
        if style == "upper":
            return val.upper()
        elif style == "lower":
            return val.lower()
        elif style == "leading_space":
            return f"  {val}"
        elif style == "trailing_space":
            return f"{val}  "
        elif style == "double_space":
            return val.replace(" ", "  ")
        return val

    return series.apply(mess)


def duplicate_random_rows(df: pd.DataFrame, rate: float, protected_ids: set, id_column: str) -> pd.DataFrame:
    """Duplicates a random fraction of rows (excluding protected/fraud-linked ones)."""
    eligible = df[~df[id_column].isin(protected_ids)] if id_column in df.columns else df
    n_dupes = int(len(df) * rate)
    if n_dupes == 0 or eligible.empty:
        return df
    dupes = eligible.sample(n=min(n_dupes, len(eligible)), replace=False)
    return pd.concat([df, dupes], ignore_index=True)


# ---------------------------------------------------------------------------
# Per-file messiness logic
# ---------------------------------------------------------------------------

def mess_beneficiaries(df: pd.DataFrame, missing_rate: float, outlier_rate: float, protected_ids: set) -> pd.DataFrame:
    df = df.copy()

    # Missing values in non-key descriptive fields
    df["phone_number"] = randomly_blank(df["phone_number"], missing_rate)
    df["address_line"] = randomly_blank(df["address_line"], missing_rate * 0.7)
    df["monthly_income_pkr"] = randomly_blank(df["monthly_income_pkr"], missing_rate * 0.5)

    # Inconsistent casing / whitespace in text fields
    df["full_name"] = randomly_whitespace_and_case(df["full_name"], 0.20)
    df["city"] = randomly_whitespace_and_case(df["city"], 0.15)

    # Inconsistent gender labels (same meaning, different representation)
    gender_variants = {"Male": ["Male", "male", "M", "MALE"], "Female": ["Female", "female", "F", "FEMALE"]}
    df["gender"] = df["gender"].apply(
        lambda g: random.choice(gender_variants.get(g, [g])) if random.random() < 0.25 else g
    )

    # Negative / invalid household size (data entry errors)
    n_bad_household = int(len(df) * outlier_rate)
    bad_idx = df.sample(n=min(n_bad_household, len(df))).index
    df.loc[bad_idx, "household_size"] = df.loc[bad_idx, "household_size"].apply(
        lambda x: random.choice([-1, 0, 99])
    )

    # Income outliers (unrealistically high, or negative from a refund/error)
    n_income_outliers = int(len(df) * outlier_rate)
    outlier_idx = df.sample(n=min(n_income_outliers, len(df))).index
    df.loc[outlier_idx, "monthly_income_pkr"] = df.loc[outlier_idx, "monthly_income_pkr"].apply(
        lambda x: random.choice([-5000, 999999])
    )

    # Mixed date formats for date_of_birth (some DD/MM/YYYY instead of YYYY-MM-DD)
    def mess_date(val):
        if pd.isna(val) or random.random() > 0.15:
            return val
        try:
            y, m, d = val.split("-")
            return f"{d}/{m}/{y}"
        except Exception:
            return val

    df["date_of_birth"] = df["date_of_birth"].apply(mess_date)

    # Inconsistent status capitalization/typos
    status_typos = {"Active": ["Active", "active", "ACTIVE", "Actve"], "Inactive": ["Inactive", "inactive"], "Suspended": ["Suspended", "suspended"]}
    df["status"] = df["status"].apply(
        lambda s: random.choice(status_typos.get(s, [s])) if random.random() < 0.15 else s
    )

    # A few fully duplicated rows (excluding fraud-linked beneficiaries)
    df = duplicate_random_rows(df, rate=0.01, protected_ids=protected_ids, id_column="beneficiary_id")

    return df


def mess_national_ids(df: pd.DataFrame, missing_rate: float) -> pd.DataFrame:
    df = df.copy()

    df["address"] = randomly_blank(df["address"], missing_rate * 0.6)
    df["full_name"] = randomly_whitespace_and_case(df["full_name"], 0.15)

    # Occasionally blank out id_status
    df["id_status"] = randomly_blank(df["id_status"], missing_rate * 0.3)

    # Mixed date formats for date_of_birth, same pattern as beneficiaries
    def mess_date(val):
        if pd.isna(val) or random.random() > 0.15:
            return val
        try:
            y, m, d = val.split("-")
            return f"{d}-{m}-{y}"
        except Exception:
            return val

    df["date_of_birth"] = df["date_of_birth"].apply(mess_date)

    return df


def mess_disbursements(df: pd.DataFrame, missing_rate: float, outlier_rate: float, protected_ids: set) -> pd.DataFrame:
    df = df.copy()

    # Missing disbursement dates / payment methods
    df["disbursement_date"] = randomly_blank(df["disbursement_date"], missing_rate * 0.5)
    df["payment_method"] = randomly_blank(df["payment_method"], missing_rate * 0.3)

    # Amount stored inconsistently: sometimes as a string with commas or a currency symbol
    def mess_amount(val):
        r = random.random()
        if r < 0.08:
            return f"Rs. {val:,}"
        elif r < 0.14:
            return f"{val:,}"
        return val

    df["amount_pkr"] = df["amount_pkr"].apply(mess_amount)

    # Negative amounts (refunds/reversal entry errors) and extreme outliers
    n_outliers = int(len(df) * outlier_rate)
    outlier_idx = df.sample(n=min(n_outliers, len(df))).index

    def force_outlier(val):
        try:
            base = float(str(val).replace("Rs.", "").replace(",", "").strip())
        except ValueError:
            base = 5000
        return random.choice([-base, base * 20])

    df.loc[outlier_idx, "amount_pkr"] = df.loc[outlier_idx, "amount_pkr"].apply(force_outlier)

    # Status typos/inconsistency
    status_variants = {"Success": ["Success", "success", "SUCCESS", "Succes"], "Pending": ["Pending", "pending"], "Reversed": ["Reversed", "reversed"]}
    df["status"] = df["status"].apply(
        lambda s: random.choice(status_variants.get(s, [s])) if random.random() < 0.15 else s
    )

    # A few duplicate disbursement rows (excluding fraud-linked beneficiaries)
    df = duplicate_random_rows(df, rate=0.015, protected_ids=protected_ids, id_column="beneficiary_id")

    return df


def main():
    parser = argparse.ArgumentParser(description="Inject realistic messiness into the raw generated CSVs.")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--ground-truth-file", type=str, default="data/ground_truth_fraud_ids.csv")
    parser.add_argument("--missing-rate", type=float, default=0.06, help="Base rate for missing values.")
    parser.add_argument("--outlier-rate", type=float, default=0.02, help="Base rate for negative/outlier numeric values.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--in-place", action="store_true", default=True,
                         help="Overwrite the same files in data/raw/ (default behavior).")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    raw_dir = Path(args.raw_dir)
    protected_ids = load_ground_truth_keys(Path(args.ground_truth_file))
    print(f"Protecting {len(protected_ids)} fraud-linked beneficiary IDs from row drop/duplication side-effects.")

    # Beneficiaries
    b_path = raw_dir / "beneficiaries.csv"
    beneficiaries = pd.read_csv(b_path)
    before = len(beneficiaries)
    beneficiaries = mess_beneficiaries(beneficiaries, args.missing_rate, args.outlier_rate, protected_ids)
    beneficiaries.to_csv(b_path, index=False)
    print(f"beneficiaries.csv: {before} -> {len(beneficiaries)} rows (messiness injected, saved to {b_path})")

    # National ID records
    n_path = raw_dir / "national_id_records.csv"
    national_ids = pd.read_csv(n_path)
    national_ids = mess_national_ids(national_ids, args.missing_rate)
    national_ids.to_csv(n_path, index=False)
    print(f"national_id_records.csv: messiness injected, saved to {n_path}")

    # Disbursements
    d_path = raw_dir / "disbursements.csv"
    disbursements = pd.read_csv(d_path)
    before = len(disbursements)
    disbursements = mess_disbursements(disbursements, args.missing_rate, args.outlier_rate, protected_ids)
    disbursements.to_csv(d_path, index=False)
    print(f"disbursements.csv: {before} -> {len(disbursements)} rows (messiness injected, saved to {d_path})")

    print("\nDone. Your raw data now contains realistic data-quality issues to handle in Stage 3.")
    print("Re-run your exploration notebook to see the new problems show up.")


if __name__ == "__main__":
    main()