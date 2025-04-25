from elsai_core.config.loggerConfig import setup_logger
import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class AzureOpenAIConnector:

    def __init__(self):
        self.logger = setup_logger()
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", None)
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", None)  
        self.openai_api_version = os.getenv("OPENAI_API_VERSION", None)
        self.temperature = float(os.getenv("AZURE_OPENAI_TEMPERATURE", 0.1))

    def connect_azure_open_ai(self, deploymentname: str):
        """
        Connects to the Azure OpenAI API using the provided model name.

        Args:
            deploymentname (str): The name of the OpenAI model to use.

        Raises:
            ValueError: If the endpoint, API key, or model name is missing.
        """

        if not self.openai_api_key:
            self.logger.error("Azure OpenAI access key is not set in the environment variables.")
            raise ValueError("Azure OpenAI Access key is missing.")
        
        if not self.azure_endpoint:
            self.logger.error("Azure OpenAI api base is not set in the environment variables.")
            raise ValueError("Azure OpenAI api base is missing.")
        
        if not self.openai_api_version:
            self.logger.error("Azure version is not set in the environment variables.")
            raise ValueError("Azure version is missing.")

        if not deploymentname:
            self.logger.error("Model name is not provided.")
            raise ValueError("Model name is missing.")

        try:
            llm = AzureChatOpenAI(
                    deployment_name=deploymentname,
                    openai_api_key=self.openai_api_key,
                    azure_endpoint=self.azure_endpoint,  
                    openai_api_version=self.openai_api_version,
                    temperature=self.temperature
                )
            self.logger.info(f"Successfully connected to Azure OpenAI model: {llm}")
            return llm
        except Exception as e:
            self.logger.error(f"Error connecting to Azure OpenAI: {e}")
            raise


