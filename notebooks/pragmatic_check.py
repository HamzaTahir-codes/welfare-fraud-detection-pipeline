""" 
Project: Welfare Fraud Detection Pipeline
This module contains the schema definitions for the database tables used in the welfare fraud detection pipeline.
The schema includes tables for beneficiaries, national IDs, and disbursements, along with their respective fields and constraints.
The schema is designed to support the extraction, transformation, and loading (ETL) processes of the pipeline, enabling efficient data storage and retrieval for fraud detection analysis.
The schema is defined using SQL statements, which can be executed to create the necessary tables in the PostgreSQL database.
"""

import sys
import os

# Get the absolute path of the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the root directory (one level up)
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from src.database.connection import get_connection

csv_to_table ={
    "data/clean/beneficiaries_cleaned_20260905011857.csv": "beneficiaries",
    "data/clean/national_id_cleaned_20260905011857.csv": "national_id_records",
    "data/clean/disbursements_cleaned_20260905011857.csv": "disbursements",
}

con = get_connection()
cur = con.cursor()

for csv_path, table_name in csv_to_table.items():
    csv_cols = set(pd.read_csv(csv_path, nrows=0).columns)
    
    

    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table_name,))
    db_cols = set(row[0] for row in cur.fetchall())
    missing_cols = csv_cols - db_cols
    if missing_cols:
        print(f"Warning: The following columns in {csv_path} are missing in the database table {table_name}: {missing_cols}")
    extra_cols = db_cols - csv_cols

if extra_cols:
    print(f"Warning: The following columns in the database table {table_name} are not present in the CSV file {csv_path}: {extra_cols}")

