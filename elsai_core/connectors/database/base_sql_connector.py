"""
This module provides a base SQL connector class for interacting with SQL databases.
"""   
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from .dialects import Dialects

class BaseSQLConnector:
    """
    A base class for SQL database connectors.
    """
    def __init__(
            self,
            dialect: str,
            llm,
            database_name: str,
            database_url: str="",
            database_user: str="",
            database_password: str="",
            driver_name: str = None
    ):
        """
        Initializes the SQL connector with the given parameters.
        """
        self.llm = llm
        self.dialect = dialect
        self.database_name = database_name
        self.database_url = database_url
        self.database_user = database_user
        self.database_password = database_password
        self.driver_name = driver_name
        if self.dialect == Dialects.ODBCMYSQL.value or self.dialect == Dialects.ODBCPOSTGRES.value:
            self.db_connection_string = f"{self.dialect}://{self.database_user}:{self.database_password}@{self.database_url}/{self.database_name}?driver={self.driver_name}"
        elif self.dialect == Dialects.SQLITE.value:
            self.db_connection_string = f"{self.dialect}:///{self.database_name}"
        else:
            self.db_connection_string = f"{self.dialect}://{self.database_user}:{self.database_password}@{self.database_url}/{self.database_name}"
        
        self.db = SQLDatabase.from_uri(self.db_connection_string)
        self.agent_executor = create_sql_agent(llm=self.llm, db=self.db, agent_type="openai-tools")

    def invoke(self, query: str):
        result = self.agent_executor.invoke(query)
        return result['output']