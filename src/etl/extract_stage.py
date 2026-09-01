'''
extract_stage.py

This module contains the ExtractStage class, which is responsible 
for extracting data from various sources and preparing it for further processing in the ETL pipeline.

'''

from posixpath import basename

import pandas as pd
import json
import logging
import os
import datetime as dt

# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_COLS_BENEFICIARIES = {
    "core": {"beneficiary_id", "cnic", "full_name"},
    "optional": {
        "father_or_husband_name", "gender", "date_of_birth",
        "phone_number", "marital_status", "household_size",
        "monthly_income_pkr", "address_line", "city", "province",
        "bank_account_number", "welfare_program", "registration_date",
        "registration_channel", "status"
    }
}

EXPECTED_COLS_DISBURSEMENTS = {
    "core": {"disbursement_id", "beneficiary_id", "amount_pkr"},
    "optional": {
        "bank_account_number", "disbursement_cycle", "disbursement_date",
        "payment_method", "branch_or_agent_code", "status"
    }
}

EXPECTED_COLS_NATIONAL_ID = {
    "core": {"national_id_no", "full_name"},
    "optional": {
        "gender", "date_of_birth", "address",
        "id_issue_date", "id_status", "linked_beneficiary_id"
    }
}

# ============================================================
# STEP 1: Loaders
# ============================================================

def load_beneficiaries(file_path: str):
    """
    Load beneficiaries data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing beneficiaries data.
    """
    try:
        beneficiaries_df = pd.read_csv(file_path)
        logging.info(f"Loaded beneficiaries data from {file_path} with shape {beneficiaries_df.shape}.")
        return beneficiaries_df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading beneficiaries data from {file_path}: {e}")
        raise
    
def load_disbursements(file_path: str):
    """
    Load disbursements data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing disbursements data.
    """
    try:
        disbursements_df = pd.read_csv(file_path)
        logging.info(f"Loaded disbursements data from {file_path} with shape {disbursements_df.shape}.")
        return disbursements_df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading disbursements data from {file_path}: {e}")
        raise
    
def load_national_id(file_path: str):
    """
    Load national ID data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing national ID data.
    """
    try:
        national_id_df = pd.read_csv(file_path)
        logging.info(f"Loaded national ID data from {file_path} with shape {national_id_df.shape}.")
        return national_id_df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading national ID data from {file_path}: {e}")
        raise
    
# ============================================================
# STEP 2: Schema validation (core vs optional)
# ============================================================

def validate_schema(df: pd.DataFrame, expected_cols: dict, source_name: str):
    """
    Validate the schema of the DataFrame against expected columns.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        expected_cols (dict): Dictionary containing 'core' and 'optional' columns.
        source_name (str): Name of the data source for logging purposes.
    """
    all_expected_cols = expected_cols["core"] | expected_cols["optional"]
    actual_cols = set(df.columns)
    
    missing_core_cols = expected_cols["core"] - actual_cols
    missing_optional_cols = expected_cols["optional"] - actual_cols
    extra_cols = actual_cols - all_expected_cols
    has_duplictaes = (len(df.columns) != len(set(df.columns)))
    
    result = {
        "is_valid" : len(missing_core_cols) == 0,
        "missing_core_cols" : missing_core_cols,
        "missing_optional_cols" : missing_optional_cols,
        "extra_cols" : extra_cols,
        "has_duplicates" : has_duplictaes
    }
    
    if missing_core_cols:
        logging.error(f"{source_name} is missing core columns: {missing_core_cols}")
    if missing_optional_cols:
        logging.warning(f"{source_name} is missing optional columns: {missing_optional_cols}")
    if extra_cols:
        logging.warning(f"{source_name} has extra columns: {extra_cols}")
    if has_duplictaes:
        logging.error(f"{source_name} has duplicate columns.")
        
    return result

# ============================================================
# STEP 3: Row-count & sanity logging
# ============================================================

def log_basic_stats(df, source_name):
    """
    Log basic statistics about the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to log stats for.
        source_name (str): Name of the data source for logging purposes.
    """
    total_rows = len(df)
    empty_rows = int(df.isnull().all(axis=1).sum())
    null_counts_per_column = df.isnull().sum().to_dict()
    for key, value in null_counts_per_column.items():
        null_counts_per_column[key] = int(value)  # Convert to int for JSON serialization
    
    logging.info(f"{source_name} - Total rows: {total_rows}, Empty rows: {empty_rows}")
    logging.info(f"{source_name} - Null counts per column: {null_counts_per_column}")
    
    return {
        "total_rows": total_rows,
        "empty_rows": empty_rows,
        "null_counts_per_column": null_counts_per_column
        }

# ============================================================
# STEP 4: Raw metadata capture
# ============================================================

def build_metadata(filepath, df, source_name, stats, status="LOADED"):
    """
    Build metadata for the extracted data.

    Args:
        filepath (str): Path to the source file.
        df (pd.DataFrame): DataFrame containing the extracted data.
        source_name (str): Name of the data source.
        stats (dict): Basic statistics about the DataFrame.
        status (str): Status of the extraction process. Default is "LOADED".
    """
    metadata = {
        "source_name": source_name,
        "file_name": basename(filepath),
        "extraction_timestamp": dt.datetime.now().isoformat(),
        "status": status,    # LOADED / SKIPPED / MISSING_FILE
        "total_rows": stats["total_rows"],
        "empty_rows": stats["empty_rows"],
        "column_count" : len(df.columns),
    }
    
    logging.info(f"Metadata for {source_name}: {json.dumps(metadata, indent=2)}")
    
    return metadata

# ============================================================
# STEP 5: Staging write
# ============================================================

def write_to_staging(df, source_name, run_id):
    """
    Write the DataFrame to a staging area.

    Args:
        df (pd.DataFrame): DataFrame to write.
        source_name (str): Name of the data source.
        run_id (str): Unique identifier for the ETL run.
    """
    output_path = "data/staging/" + source_name + "_" + run_id + ".csv"
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Wrote {source_name} data to staging at {output_path}.")
    except Exception as e:
        logging.error(f"Error writing {source_name} data to staging at {output_path}: {e}")
        raise
    
    return output_path

# ============================================================
# STEP 6: Orchestrator (hub-and-spoke failure behavior)
# ============================================================

def run_extract():
    """
    Orchestrate the extraction process for all data sources.
    """
    run_id = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    all_metadata = []
    run_status = "SUCCESS"
    
    # ---------- HUB: beneficiaries ----------
    try:
        df_beneficiaries = load_beneficiaries("data/raw/beneficiaries.csv")
    except FileNotFoundError:
        logging.error("Beneficiaries file not found. Skipping extraction for beneficiaries.")
        return {
            "run_id": run_id,
            "status": "FAILED",
            "message": "Beneficiaries file not found.",
            "sources" : []
        }
    result_beneficiaries = validate_schema(df_beneficiaries, EXPECTED_COLS_BENEFICIARIES, "beneficiaries")
    
    if result_beneficiaries["is_valid"] == False:
        logging.error("Beneficiaries schema validation failed. Skipping extraction for beneficiaries.")
        return {
            "run_id": run_id,
            "status": "FAILED",
            "message": "Beneficiaries schema validation failed.",
            "validation_result": result_beneficiaries,
            "sources" : []
        }
    stats_beneficiaries = log_basic_stats(df_beneficiaries, "beneficiaries")
    metadata_beneficiaries = build_metadata("data/raw/beneficiaries.csv", df_beneficiaries, "beneficiaries", stats_beneficiaries)
    write_to_staging(df_beneficiaries, "beneficiaries", run_id)
    all_metadata.append(metadata_beneficiaries)
    
    # ---------- SPOKES: disbursements & national_id ----------
    spokes = [
        ("disbursements", load_disbursements, "data/raw/disbursements.csv", EXPECTED_COLS_DISBURSEMENTS),
        ("national_id", load_national_id, "data/raw/national_id_records.csv", EXPECTED_COLS_NATIONAL_ID)
    ]
    
    skipped_sources = [] # Keep track of skipped sources due to errors, for transform to check
    for (name, loader_func, file_path, expected_cols) in spokes:
        try:
            df = loader_func(file_path)
        except FileNotFoundError:
            logging.error(f"{name} file not found. Skipping extraction for {name}.")
            all_metadata.append({"source_name": name, "status": "MISSING_FILE"})
            skipped_sources.append(name)
            run_status = "PARTIAL_SUCCESS"
            continue
        
        result = validate_schema(df, expected_cols, name)
        if result["is_valid"] == False:
            logging.error(f"{name} schema validation failed. Skipping extraction for {name}.")
            all_metadata.append({"source_name": name, "status": "SKIPPED", "validation_result": result})
            skipped_sources.append(name)
            run_status = "PARTIAL_SUCCESS"
            continue
        
        stats = log_basic_stats(df, name)
        metadata = build_metadata(file_path, df, name, stats)
        write_to_staging(df, name, run_id)
        all_metadata.append(metadata)
        
    # ---------- Final summary ----------
    summary = {
        "run_id": run_id,
        "status": run_status,
        "skipped_sources": skipped_sources,
        "sources": all_metadata
    }
    
    # open file to write summary to JSON
    summary_file_path = f"data/staging/extract_run_{run_id}_metadata.json"
    try:
        with open(summary_file_path, "w") as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Extraction summary written to {summary_file_path}.")
    except Exception as e:
        logging.error(f"Error writing extraction summary to {summary_file_path}: {e}")
        raise
    
    logging.info(f"Extraction run {run_id} completed with status: {run_status}. Metadata written to data/staging/extract_run_{run_id}_metadata.json")
    return summary

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_extract()