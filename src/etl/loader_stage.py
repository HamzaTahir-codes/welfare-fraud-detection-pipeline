""" 
Loader Stage Module
This module is responsible for loading the cleaned and transformed data into the database. It handles the insertion of data into the appropriate tables, ensuring that the data integrity is maintained and that any necessary constraints are respected.
The loader stage is a critical part of the ETL (Extract, Transform, Load) pipeline as it finalizes the data processing by persisting the cleaned data into the database for further analysis and reporting.
"""

import logging
import sys
import os
import pandas as pd

# Get the absolute path of the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the root directory (one level up)
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from psycopg2.extras import execute_values
from src.database.connection import get_connection

def clean_flags_column(df, column="data_quality_flags"):
    """
    Ensures that the 'data_quality_flags' column is a list of strings for each row.
    If the column is a string representation of a list, it converts it to an actual list.
    If the column is missing or not in the expected format, it initializes it as an empty list.
    """
    if column not in df.columns:
        df[column] = [[] for _ in range(len(df))]
    else:
        # Convert string representations of lists to actual lists
        df[column] = df[column].apply(
            lambda x: [] if pd.isna(x) else [flag.strip() for flag in x.strip("[]").split(", ") if flag.strip()] if isinstance(x, str) else x
        )
    
    return df

def prepare_records(df, columns):
    """
    Prepares the records for insertion into the database.
    Ensures that the DataFrame contains only the specified columns and that the 'data_quality_flags' column is a list of strings.
    Returns a list of tuples representing the records to be inserted.
    """
    df = clean_flags_column(df, column="data_quality_flags")
    
    # Ensure only the specified columns are present
    df = df[columns]
    
    # Convert DataFrame to list of tuples, replacing NaN with None
    records = [
        tuple(
            val if isinstance(val, list) else (None if pd.isna(val) else val)
            for val in row
        )
        for row in df.to_numpy()
    ]
    return records
    
def load_dataframe(df, table_name, columns, conn):
    """_summary_

    Args:
        df (_type_): _description_
        table_name (_type_): _description_
        columns (_type_): _description_
        conn (_type_): _description_
        
    Loads one DataFrame into one table using execute_values (batched insert).
    Raises the underlying exception on failure -- caller decides what
    halting/PARTIAL behavior that failure means for the overall run.
    """
    
    records = prepare_records(df, columns)
    cursor = conn.cursor()
    
    # Build column list string and %s placeholder from columns
    column_list = ", ".join(columns)
    
    insert_query = f"INSERT INTO {table_name} ({column_list}) VALUES %s"
    
    execute_values(cursor, insert_query, records)
    
    conn.commit()  # Commit the transaction after successful insertion
    
    cursor.close()  # Close the cursor after operation
        
        
def loader_stage():
    """
    Orchestrates loading all three source tables.
    Returns a structured result dict, same pattern as run_extract() in Stage 2 --
    keeps your pipeline's reporting style consistent across stages.
    """
    result = {
        "status": None,
        "tables_loaded": [],
        "tables_failed": [],
        "errors": {}
    }
    
    conn = get_connection()
    
    try:
        # Hub table -- beneficiaries --
        beneficiaries_df = pd.read_csv("data/clean/beneficiaries_cleaned.csv")
        
        # columns 
        beneficiaries_columns = [
                "beneficiary_id",
                "full_name",
                "father_or_husband_name",
                "gender",
                "date_of_birth",
                "cnic",
                "phone_number",
                "marital_status",
                "household_size",
                "monthly_income_pkr",
                "address_line",
                "city",
                "province",
                "bank_account_number",
                "welfare_program",
                "registration_date",
                "registration_channel",
                "status",
                "gender_clean",
                "status_clean",
                "date_of_birth_clean",
                "dob_parse_failed",
                "registration_date_clean",
                "reg_date_parse_failed",
                "monthly_income_clean",
                "income_flag",
                "phone_missing",
                "address_missing",
                "is_duplicate_row",
                "data_quality_flags"
            ]
        try:
            load_dataframe(beneficiaries_df, "beneficiaries", beneficiaries_columns, conn)
            result["tables_loaded"].append("beneficiaries")
        except Exception as e:
            result['status'] = 'FAILED'
            result["tables_failed"].append("beneficiaries")
            result["errors"]["beneficiaries"] = str(e)
            conn.rollback()  # Rollback the transaction on failure
            return result  # Stop further processing if one table fails
        
        # spot table -- national_id_records --
        try:
            national_id_df = pd.read_csv("data/clean/national_id_cleaned.csv")

            # columns
            national_id_columns = [
                    "national_id_no",
                    "full_name",
                    "gender",
                    "date_of_birth",
                    "address",
                    "id_issue_date",
                    "id_status",
                    "linked_beneficiary_id",
                    "gender_clean",
                    "date_of_birth_clean",
                    "dob_parse_failed",
                    "id_issue_date_clean",
                    "issue_date_parse_failed",
                    "address_missing",
                    "id_status_missing",
                    "no_linked_beneficiary",
                    "data_quality_flags"
                ]
            load_dataframe(national_id_df, "national_id_records", national_id_columns, conn)
            result["tables_loaded"].append("national_id_records")
        except Exception as e:
            result["tables_failed"].append("national_id_records")
            result["errors"]["national_id_records"] = str(e)
            conn.rollback()  # Rollback the transaction on failure
        
        # spot table -- disbursements --
        try:
            disbursements_df = pd.read_csv("data/clean/disbursements_cleaned.csv")
            
            # columns
            disbursements_columns = [
                "disbursement_id",
                "beneficiary_id",
                "bank_account_number",
                "amount_pkr",
                "disbursement_cycle",
                "disbursement_date",
                "payment_method",
                "branch_or_agent_code",
                "status",
                "status_clean",
                "disbursement_date_clean",
                "disb_date_parse_failed",
                "amount_pkr_clean",
                "amount_flag",
                "payment_method_missing",
                "data_quality_flags",
                "shared_bank_account_flag",
                "orphaned_beneficiary_flag"
            ]
            load_dataframe(disbursements_df, "disbursements", disbursements_columns, conn)
            result["tables_loaded"].append("disbursements")
        except Exception as e:
            result["tables_failed"].append("disbursements")
            result["errors"]["disbursements"] = str(e)
            conn.rollback()  # Rollback the transaction on failure
        
        if result["tables_failed"]:
            result['status'] = 'PARTIAL'
        else:
            result['status'] = 'SUCCESS'
            
        return result
    
    except Exception as e:
        result['status'] = 'FAILED'
        result["errors"]["general"] = str(e)
        conn.rollback()  # Rollback the transaction on failure
        return result
    
    finally:
        conn.close()  # Ensure the connection is closed after operation
        
if __name__ == "__main__":
    loader_result = loader_stage()
    print(loader_result)