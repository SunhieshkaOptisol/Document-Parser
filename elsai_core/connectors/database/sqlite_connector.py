import os
from .dialects import Dialects
from .base_sql_connector import BaseSQLConnector
class SQLiteConnector(BaseSQLConnector):
    """
    A connector class for sqlite3 databases.
    """

    def __init__(self, llm, database_path:str = os.getenv("DB_NAME")):
        super().__init__(
    Dialects.SQLITE.value, llm=llm, database_name=database_path
        )
        