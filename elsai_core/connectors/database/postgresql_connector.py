from .base_sql_connector import BaseSQLConnector
from .dialects import Dialects
import os

class PostgreSQLConnector(BaseSQLConnector):
    """
    A connector class for PostgreSQL databases.
    """
    def __init__(
            self, 
            llm,
            database_name: str = os.getenv("DB_NAME"),
            database_url: str = os.getenv("DB_URL"),
            database_user: str = os.getenv("DB_USER"),
            database_password: str = os.getenv("DB_PASSWORD")
        ):
        super().__init__(
            Dialects.POSTGRES.value, llm, database_name, 
            database_url, database_user, database_password
        )
