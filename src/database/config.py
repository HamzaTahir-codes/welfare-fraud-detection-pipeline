""" 
Database configuration module for the welfare fraud detection pipeline.

This module provides the necessary configuration for connecting to the PostgreSQL database used in the welfare fraud detection pipeline. It reads database connection parameters from environment variables and constructs a connection string that can be used by SQLAlchemy or other database libraries.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def get_db_config():
    """
    Reads DB credentials from environment, returns as a dict.
    Raises an error if any required variable is missing —
    fail loudly here rather than silently connecting to the wrong thing.
    """
    required_vars = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing_vars = [var for var in required_vars if var not in os.environ]
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return {
        "host": os.environ["DB_HOST"],
        "port": os.environ["DB_PORT"],
        "database": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"]
    }