import os
import boto3
from elsai_core.config.loggerConfig import setup_logger
from langchain_community.document_loaders import AmazonTextractPDFLoader
from ..connectors.aws_s3 import AwsS3Connector

class AwsTextractConnector:
    """
    A class to extract text from PDF files using AWS Textract after uploading to S3.
    It handles authentication, file upload, text extraction, and cleanup in AWS S3.
    """
    def __init__(self, access_key: str = None,
                 secret_key: str = None, session_token: str = None, region_name: str = "us-east-1"):
        self.textract_client = boto3.client(
            "textract",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
            region_name=region_name
        )
        self.s3_connector = AwsS3Connector(access_key, secret_key, session_token)
        self.logger = setup_logger()
        self.s3_folder = os.getenv("S3_FOLDER")
        self.s3_bucket = os.getenv("S3_BUCKET")

    def extract_text(self, file_path: str):
        """
        Large files must be uploaded to S3 before extracting text using AWS Textract
        """
        file_name = os.path.basename(file_path)
        s3_key = f"{self.s3_folder}/{file_name}"
        s3_uri = self.s3_connector.upload_file_to_s3(self.s3_bucket, s3_key, file_path)
        try:
            self.logger.info("Extracting text from %s using AWS Textract", s3_uri)
            loader = AmazonTextractPDFLoader(s3_uri, client=self.textract_client)
            documents =  loader.load()
            return documents
        except Exception as e:
            self.logger.error("Error extracting text: %s", e)
            raise e
        finally:
            self.s3_connector.delete_file_from_s3(self.s3_bucket, s3_key)
