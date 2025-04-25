"""
This module provides a connector for AWS S3 using boto3.
"""

import os
import boto3
from elsai_core.config.loggerConfig import setup_logger

class AwsS3Connector:
    """
    A connector class for interacting with AWS S3.
    """

    def __init__(self, access_key: str = None, secret_key: str = None, session_token: str = None):
        """
        Initializes the S3Connector with AWS credentials.

        :param access_key: AWS access key ID
        :param secret_key: AWS secret access key
        :param session_token: AWS session token
        """
        self.logger = setup_logger()
        self.access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID", None)
        self.secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY", None)
        self.session_token = session_token or os.getenv("AWS_SESSION_TOKEN", None)
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            aws_session_token=self.session_token
        )

    def upload_file_to_s3(self, bucket_name: str, s3_key: str, file_path: str):
        """
        Uploads a file to an S3 bucket.
        """
        try:
            self.s3.upload_file(file_path, bucket_name, s3_key)
            self.logger.info("File %s uploaded successfully to %s", file_path, s3_key)
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            return s3_uri
        except Exception as e:
            self.logger.error("Error uploading file: %s", e)
            raise e
           
    def delete_file_from_s3(self, bucket_name: str, s3_key: str):
        """
        Deletes a file from an S3 bucket.
        """
        try:
            self.s3.delete_object(Bucket=bucket_name, Key=s3_key)
            self.logger.info("File %s deleted successfully from %s", s3_key, bucket_name)
        except Exception as e:
            self.logger.error("Error deleting file: %s", e)
            raise e

    def download_file_from_s3(self, bucket_name: str, file_name: str, download_path: str):
        """
        Downloads a file from an S3 bucket to a specified local path.

        :param bucket_name: Name of the S3 bucket
        :param file_name: Name of the file to download
        :param download_path: Local path to download the file to
        """
        raw_file_name = os.path.basename(file_name)
        download_path = os.path.join(download_path, raw_file_name)
        try:
            self.s3.download_file(bucket_name, file_name, download_path)
            self.logger.info("File %s downloaded successfully to %s", file_name, download_path)
        except Exception as e:
            self.logger.error("Error downloading file: %s", e)
            raise e
