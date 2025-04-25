from elsai_core.config.loggerConfig import setup_logger
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class OpenAIConnector:

    def __init__(self):
        self.logger = setup_logger()
        self.access_key = os.getenv("OPENAI_API_KEY", None)
        

    def connect_open_ai(self, modelname: str="gpt-4o-mini"):
        """
        Connects to the OpenAI API using the provided model name.

        Args:
            modelname (str): The name of the OpenAI model to use.

        Raises:
            ValueError: If the access key or model name is missing.
        """
        if not self.access_key:
            self.logger.error("OpenAI access key is not set in the environment variables.")
            raise ValueError("Access key is missing.")

        if not modelname:
            self.logger.error("Model name is not provided.")
            raise ValueError("Model name is missing.")

        try:
            llm = ChatOpenAI(
                openai_api_key = self.access_key, 
                model_name= modelname, 
            )
            self.logger.info(f"Successfully connected to OpenAI model: {llm}")
            return llm
        except Exception as e:
            self.logger.error(f"Error connecting to OpenAI: {e}")
            raise