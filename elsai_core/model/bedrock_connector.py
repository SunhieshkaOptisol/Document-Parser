from elsai_core.config.loggerConfig import setup_logger
import os
from langchain_aws import BedrockLLM
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class BedrockConnector:

    def __init__(self):
        self.logger = setup_logger()
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", None)
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", None)
        self.aws_region = os.getenv("AWS_REGION", None)
        self.temperature = float(os.getenv("BEDROCK_TEMPERATURE", 0.1))

    def connect_bedrock(self, model_id: str):
        """
        Connects to the AWS Bedrock API using the provided model ID.

        Args:
            model_id (str): The ID of the Bedrock model to use (e.g., 'anthropic.claude-v2', 'amazon.titan-text-express-v1').

        Raises:
            ValueError: If the AWS credentials, region, or model ID is missing.
        """

        if not self.aws_access_key:
            self.logger.error("AWS access key ID is not set in the environment variables.")
            raise ValueError("AWS access key ID is missing.")
        
        if not self.aws_secret_key:
            self.logger.error("AWS secret access key is not set in the environment variables.")
            raise ValueError("AWS secret access key is missing.")
        
        if not self.aws_region:
            self.logger.error("AWS region is not set in the environment variables.")
            raise ValueError("AWS region is missing.")

        if not model_id:
            self.logger.error("Model ID is not provided.")
            raise ValueError("Model ID is missing.")

        try:
            llm = BedrockLLM(credentials_profile_name="bedrock-admin", model_id=model_id)
            self.logger.info(f"Successfully connected to AWS Bedrock model: {model_id}")
            return llm
        except Exception as e:
            self.logger.error(f"Error connecting to AWS Bedrock: {e}")
            raise