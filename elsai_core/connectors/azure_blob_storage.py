"""
This module provides a class for interacting with Azure Blob Storage.
"""

import os
from azure.storage.blob import BlobServiceClient
from elsai_core.config.loggerConfig import setup_logger

class AzureBlobStorage:
    """
    A class to handle Azure Blob Storage operations.
    """
    def __init__(self, connection_string):
        """
        Initialize the AzureBlobStorage with a connection string.
        """
        self.connection_string = connection_string
        self.logger = setup_logger()
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    def download_file(self, container_name, blob_name, target_folder_path):
        """
        Download a file from Azure Blob Storage to a local directory.
        """
        file_path = os.path.join(target_folder_path, blob_name)
        blob_client = self.blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )
        try:
            self.logger.info(
                "Downloading blob '%s' to local file '%s'...", blob_name, file_path
            )
            with open(file_path, "wb") as file:
                file.write(blob_client.download_blob().readall())

            self.logger.info(
                "Blob '%s' successfully downloaded to '%s'.", blob_name, file_path
            )
        except Exception as e:  
            self.logger.error("Error: %s", e)
