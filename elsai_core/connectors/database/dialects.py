"""
This module defines the database dialects supported by the application.
"""

from enum import Enum

class Dialects(Enum):
    """
    An enumeration of supported database dialects.
    """
    MYSQL = "mysql+mysqldb"
    POSTGRES = "postgresql+psycopg2"
    ODBCPOSTGRES = "postgresql+pyodbc"
    ODBCMYSQL = "mysql+pyodbc"
    SQLITE = "sqlite"