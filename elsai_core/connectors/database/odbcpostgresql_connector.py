from .base_sql_connector import BaseSQLConnector
from .dialects import Dialects
import os

class OdbcPostgresqlConnector(BaseSQLConnector):
    """
    A connector class for ODBC PostgreSQL databases.
    """
    def __init__(
            self, 
            llm,
            database_name: str = os.getenv("DB_NAME"),
            database_url: str = os.getenv("DB_URL"),
            database_user: str = os.getenv("DB_USER"),
            database_password: str = os.getenv("DB_PASSWORD"),
            driver_name: str = os.getenv("DB_DRIVER_NAME")
        ):
        super().__init__(
            Dialects.ODBCPOSTGRES.value, llm, database_name, 
            database_url, database_user, database_password, driver_name
        )
