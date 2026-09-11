'''
transform_stage.py

This stage is responsible for transforming the data into a format suitable for analysis and modeling. 
It includes data cleaning, feature engineering, and any other necessary transformations.
It also includes the logic for handling missing values, encoding categorical variables, and scaling numerical features.
'''



import pandas as pd
import datetime as dt
import logging

from extract_stage import run_extract

# ============================================================
# CONSTANTS
# ============================================================

INCOME_CAP_PKR = 100000   # anything above this, or negative, is implausible

# Canonical mappings — lowercase/stripped input -> canonical Title Case value
STATUS_MAPPING_BENEFICIARY = {
    "active": "Active", "actve": "Active",
    "inactive": "Inactive",
    "suspended": "Suspended"
}

STATUS_MAPPING_DISBURSEMENT = {
    "success": "Success", "succes": "Success",
    "pending": "Pending",
    "reversed": "Reversed"
}

GENDER_MAPPING = {
    "f": "Female", "female": "Female",
    "m": "Male", "male": "Male"
}

# Columns that must NEVER be modified by cleaning logic
PROTECTED_COLUMNS = {
    "beneficiary_id", "cnic", "national_id_no",
    "bank_account_number", "linked_beneficiary_id", "disbursement_id"
}

# ============================================================
# STEP 1: Generic cleaning helpers (reused across all 3 files)
# ============================================================

def standardize_categorical(series, mapping_dict):
    """
    Standardizes a categorical pandas Series based on a provided mapping dictionary.
    
    Parameters:
        series (pd.Series): The input Series to standardize.
        mapping_dict (dict): A dictionary mapping lowercase/stripped values to canonical values.
        
    Returns:
        pd.Series: A new Series with standardized categorical values.
    """
    # lowercase + strip whitespace, then map to canonical value
    # anything not found in mapping is left as-is (so it's visible, not silently dropped)
    cleaned = series.str.lower().str.strip().map(mapping_dict)
    cleaned = cleaned.fillna(series)  # fill NaNs with original values for unmapped entries
    
    return cleaned

def parse_date_flexible(series):
    """
    Parses a pandas Series of date strings into datetime objects, handling multiple formats.
    
    Parameters:
        series (pd.Series): The input Series containing date strings.
        
    Returns:
        pd.Series: A new Series with parsed datetime objects.
    """
    # Try multiple known formats (DD/MM/YYYY, YYYY-MM-DD, etc.)
    # pandas.to_datetime with dayfirst inference / errors="coerce" handles most of this,
    # but you may need to try a list of explicit formats if inference is unreliable.
    parsed = pd.to_datetime(series, errors='coerce', format="mixed",dayfirst=True) # unparseable -> NaT
    formatted = parsed.dt.strftime('%Y-%m-%d')  # standardize to YYYY-MM-DD
    parse_failed_flag = parsed.isna() & series.notna()
    
    return formatted, parse_failed_flag

def strip_currency_symbols(series):
    """
    Strips common currency symbols and formatting from a pandas Series of strings.
    
    Parameters:
        series (pd.Series): The input Series containing currency strings.
    Returns:
        pd.Series: A new Series with currency symbols removed.
    """
    # remove "Rs.", commas, and any stray whitespace, leaving only digit/./- characters
    cleaned = series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)  # keep digits, dot, minus
    return cleaned

def clean_numeric_with_cap(series, cap, allow_negative=False):
    """
    Cleans a numeric pandas Series by removing implausible values and applying a cap.
    
    Parameters:
        series (pd.Series): The input Series to clean.
        cap (float): The maximum plausible value; anything above this is set to NaN.
        allow_negative (bool): Whether to allow negative values. If False, negatives are set to NaN.
        
    Returns:
        pd.Series: A new Series with cleaned numeric values.
    """
    # Only strip currency symbols from amount_pkr column
    if series.name == "amount_pkr" or series.name == "monthly_income_pkr":
        series = strip_currency_symbols(series)
    numeric = pd.to_numeric(series, errors='coerce')  # convert to numeric, unparseable -> NaN
    invalid_mask = numeric.isna() & series.notna()  # original non-NaN but unparseable
    
    if not allow_negative:
        invalid_mask |= (numeric < 0)  # mark negatives as invalid if not allowed
    invalid_mask |= (numeric > cap)  # mark values above cap as invalid
    
    cleaned_value = numeric.astype("Float64").copy()
    cleaned_value[invalid_mask] = pd.NA  # set invalid entries to NaN
    
    return cleaned_value, invalid_mask # Invalid Mask will become flag

# ============================================================
# STEP 2: Per-source cleaning functions
# ============================================================

def build_flag_summary(df, flag_columns):
    """
    Combines multiple boolean flag columns into a single summary column.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing flag columns.
        flag_columns (list): A list of column names that are boolean flags.
        
    Returns:
        pd.Series: A new Series with combined flag summaries.
    """
    # Create a summary string for each row, listing the names of flags that are True
    summary = df[flag_columns].apply(lambda row: ', '.join([col for col in flag_columns if row[col]]), axis=1)
    return summary

def clean_beneficiaries(df):
    """
    Cleans the beneficiaries DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input beneficiaries DataFrame.
        
    Returns:
        pd.DataFrame: A new DataFrame with cleaned beneficiaries data.
    """
    df = df.copy()
    data_quality_flags = []   # collect flag column names added, for logging/summary later

    df["gender_clean"] = standardize_categorical(df["gender"], GENDER_MAPPING)
    df["status_clean"] = standardize_categorical(df["status"], STATUS_MAPPING_BENEFICIARY)

    df["date_of_birth_clean"], df["dob_parse_failed"] = parse_date_flexible(df["date_of_birth"])
    df["registration_date_clean"], df["reg_date_parse_failed"] = parse_date_flexible(df["registration_date"])

    df["monthly_income_clean"], df["income_flag"] = clean_numeric_with_cap(
        df["monthly_income_pkr"], INCOME_CAP_PKR, allow_negative=False
    )

    # Missing-value flags (informational, not necessarily "bad", just noted)
    df["phone_missing"] = df["phone_number"].isna()
    df["address_missing"] = df["address_line"].isna()

    # Duplicate rows — flag fully duplicate rows (keep them, per your policy)
    df["is_duplicate_row"] = df.duplicated(keep=False)

    # Combine data-quality flags into one column (list or comma-joined string)
    df["data_quality_flags"] = build_flag_summary(df, [
        "dob_parse_failed", "reg_date_parse_failed", "income_flag",
        "phone_missing", "address_missing", "is_duplicate_row"
    ])

    return df

def clean_national_id_records(df):
    """
    Cleans the national ID records DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input national ID records DataFrame.
        
    Returns:
        pd.DataFrame: A new DataFrame with cleaned national ID records data.
    """
    df = df.copy()
    data_quality_flags = []   # collect flag column names added, for logging/summary later
    
    df["gender_clean"] = standardize_categorical(df["gender"], GENDER_MAPPING)
    df["date_of_birth_clean"], df["dob_parse_failed"] = parse_date_flexible(df["date_of_birth"])
    df["id_issue_date_clean"], df["issue_date_parse_failed"] = parse_date_flexible(df["id_issue_date"])

    df["address_missing"] = df["address"].isna()
    df["id_status_missing"] = df["id_status"].isna()

    # linked_beneficiary_id nulls are NOT cleaned/filled — that's real signal (unmatched record)
    # just note it as informational, not a "bad data" flag
    df["no_linked_beneficiary"] = df["linked_beneficiary_id"].isna()

    df["data_quality_flags"] = build_flag_summary(df, [
        "dob_parse_failed", "issue_date_parse_failed",
        "address_missing", "id_status_missing"
    ])
    
    return df

def clean_disbursements(df):
    """
    Cleans the disbursements DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input disbursements DataFrame.
        
    Returns:
        pd.DataFrame: A new DataFrame with cleaned disbursements data.
    """
    df = df.copy()
    data_quality_flags = []   # collect flag column names added, for logging/summary later

    df["status_clean"] = standardize_categorical(df["status"], STATUS_MAPPING_DISBURSEMENT)
    df["disbursement_date_clean"], df["disb_date_parse_failed"] = parse_date_flexible(df["disbursement_date"])

    # amount_pkr came in as text (object dtype) — clean_numeric_with_cap also handles
    # stripping non-numeric characters via pd.to_numeric's coercion; if that's not enough,
    # you may need a regex cleanup step first (strip currency symbols/commas) before this call
    df["amount_pkr_clean"], df["amount_flag"] = clean_numeric_with_cap(
        df["amount_pkr"], cap=10000, allow_negative=False
    )

    df["payment_method_missing"] = df["payment_method"].isna()

    df["data_quality_flags"] = build_flag_summary(df, [
        "disb_date_parse_failed", "amount_flag", "payment_method_missing"
    ])
    
    return df

# ============================================================
# STEP 3: Fraud-signal flags (kept in a SEPARATE column set)
# ============================================================

def flag_fraud_signals_disbursements(df):
    """
    Flags potential fraud signals in the disbursements DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input disbursements DataFrame.
        
    Returns:
        pd.DataFrame: A new DataFrame with fraud signal flags added.
    """
    df = df.copy()

    # Shared bank account across many disbursements/beneficiaries — planted fraud pattern
    account_counts = df["bank_account_number"].value_counts()
    suspicious_accounts = account_counts[account_counts > 3].index
    df["shared_bank_account_flag"] = df["bank_account_number"].isin(suspicious_accounts)

    return df

def flag_orphaned_disbursements(df_disbursements, df_beneficiaries):
    """
    Flags disbursement records that do not have a matching beneficiary record.
    
    Parameters:
        df_disbursements (pd.DataFrame): The input disbursements DataFrame.
        df_beneficiaries (pd.DataFrame): The input beneficiaries DataFrame.
        
    Returns:
        pd.DataFrame: A new DataFrame with orphaned disbursement flags added.
    """
    df = df_disbursements.copy()
    
    # Identify orphaned disbursements (no matching beneficiary_id)
    valid_beneficiary_ids = set(df_beneficiaries["beneficiary_id"])
    df["orphaned_beneficiary_flag"] = ~df["beneficiary_id"].isin(valid_beneficiary_ids)

    return df

def cnic_mismatch(df_beneficiaries, df_national_id):
    """
    Identifies CNIC mismatches between beneficiaries and national ID records.
    
    Parameters:
        df_beneficiaries (pd.DataFrame): The input beneficiaries DataFrame.
        df_national_id (pd.DataFrame): The input national ID records DataFrame.
        
    Returns:
        pd.DataFrame: A new DataFrame with CNIC mismatch flags added.
    """
    # Join on beneficiary_id <-> linked_beneficiary_id to compare cnic vs national_id_no
    merged = df_beneficiaries.merge(
        df_national_id,
        left_on="beneficiary_id",
        right_on="linked_beneficiary_id",
        how="left",
        suffixes=('_beneficiary', '_national')
    )

    merged["cnic_mismatch_flag"] = (merged["cnic"].notna()) & (merged["national_id_no"].notna()) & (merged["cnic"] != merged["national_id_no"])
    return merged

# ============================================================
# STEP 4: Write cleaned tables to data/clean/
# ============================================================
def write_cleaned_tables(df, source_name, run_id):
    """
    Writes the cleaned DataFrame to a CSV file in the data/clean/ directory.
    
    Parameters:
        df (pd.DataFrame): The cleaned DataFrame to write.
        source_name (str): The name of the source (e.g., 'beneficiaries', 'national_id', 'disbursements').
        run_id (str): A unique identifier for this ETL run (used in the filename).
    """
    output_path = "data/clean/" + source_name + "_" + run_id + ".csv"
    df.to_csv(output_path, index=False)
    logging.info(f"Cleaned {source_name} data written to {output_path}")
    return output_path

# ============================================================
# STEP 5: Orchestrator — respects Stage 2's extract summary
# ============================================================
def load_staged(source_name, run_id):
    """
    Loads a staged CSV file from the data/staging/ directory based on source name and run ID.
    
    Parameters:
        source_name (str): The name of the source (e.g., 'beneficiaries', 'national_id', 'disbursements').
        run_id (str): A unique identifier for this ETL run (used in the filename).
        """
    input_path = "data/staging/" + source_name + "_" + run_id + ".csv"
    try:
        df = pd.read_csv(input_path)
        logging.info(f"Loaded staged {source_name} data from {input_path}")
        return df
    except FileNotFoundError:
        logging.error(f"Staged file not found: {input_path}")
        return pd.DataFrame()  # Return empty DataFrame if file not found

def run_transform_stage(extract_summary):
    """
    Orchestrates the transformation stage of the ETL pipeline.
    
    Parameters:
        extract_summary (dict): A summary of the extracted data, including DataFrames and metadata.
        
    Returns:
        dict: A summary of the transformed data, including cleaned DataFrames and any flags.
    """
    if extract_summary["status"] == "FAILED":
        logging.error("Extract stage failed; cannot proceed to transform stage.")
        return {"status": "failed", "reason": "Extract stage failed."}
    
    run_id = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    skipped = extract_summary.get("skipped_sources", [])
    
    # Load staged files from data/staging/ written by the extract stage using extract_summary run_id
    df_beneficiaries = load_staged("beneficiaries", extract_summary["run_id"])
    df_beneficiaries_cleaned = clean_beneficiaries(df_beneficiaries)
    
    results = {"beneficiaries_cleaned": df_beneficiaries_cleaned}
    
    if "national_id" not in skipped:
        df_national_id = load_staged("national_id", extract_summary["run_id"])
        df_national_id_cleaned = clean_national_id_records(df_national_id)
        results["national_id_cleaned"] = df_national_id_cleaned
        
    else:
        logging.warning("National IDs source was skipped in extract; skipping cleaning.")
        results["national_id_cleaned"] = pd.DataFrame()  # empty DataFrame
        
    if "disbursements" not in skipped:
        df_disbursements = load_staged("disbursements", extract_summary["run_id"])
        df_disbursements_cleaned = clean_disbursements(df_disbursements)
        df_disbursements_cleaned = flag_fraud_signals_disbursements(df_disbursements_cleaned)
        df_disbursements_cleaned = flag_orphaned_disbursements(df_disbursements_cleaned, df_beneficiaries_cleaned)
        results["disbursements_cleaned"] = df_disbursements_cleaned
    else:
        logging.warning("Disbursements source was skipped in extract; skipping cleaning.")
        results["disbursements_cleaned"] = pd.DataFrame()  # empty DataFrame
        
    if not results["national_id_cleaned"].empty:
        joined_beneficiary_national_id = cnic_mismatch(df_beneficiaries_cleaned, results["national_id_cleaned"])
        results["beneficiary_national_id_joined"] = joined_beneficiary_national_id
    else:
        logging.warning("Skipping CNIC mismatch check — national_id data unavailable this run.")
    
    for name, df in results.items():
        if not df.empty:
            write_cleaned_tables(df, name, run_id)
    
    transform_status = "PARTIAL" if skipped else "success"
    return {
        "status": transform_status,
        "run_id": run_id,
        "skipped_sources": skipped,
        "cleaned_dataframes": results
    }
    
if __name__ == "__main__":
    extract_summary = run_extract()   # or load its saved JSON if run separately
    run_transform_stage(extract_summary)