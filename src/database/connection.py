"""
This module provides a function to establish a connection to the PostgreSQL database using psycopg2.
It reads the database configuration from environment variables and constructs a connection string.
The connection can be used to execute SQL queries and interact with the database.
Usage:
    from src.database.connection import get_db_connection

    conn = get_db_connection()
    # Use the connection to execute queries
    conn.close()    
"""

import psycopg2
from psycopg2 import OperationalError
from src.database.config import get_db_config

def get_connection():
    """
    Establishes a connection to the PostgreSQL database using psycopg2.
    Reads the database configuration from environment variables.
    Returns a psycopg2 connection object.
    Raises an OperationalError if the connection fails.
    """
    db_config = get_db_config()
    
    try:
        connection = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"]
        )
        return connection
    except OperationalError as e:
        raise OperationalError(f"Failed to connect to the database: {e}")