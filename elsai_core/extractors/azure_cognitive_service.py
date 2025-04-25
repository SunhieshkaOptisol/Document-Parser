import os
import time
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.core.exceptions import AzureError
from msrest.authentication import CognitiveServicesCredentials
from elsai_core.config.loggerConfig import setup_logger

class AzureCognitiveService:
    """
    A class to extract text from PDF files using Azure Cognitive Services' Read API.
    It handles authentication, text extraction, and error logging for the PDF processing.
    """
    def __init__(self, file_path:str):
        # Initialize logger
        self.logger = setup_logger()
        # Retrieve Azure credentials from environment variables
        self.subscription_key = os.environ.get("AZURE_SUBSCRIPTION_KEY")
        self.endpoint = os.environ.get("AZURE_ENDPOINT")
        self.file_path = file_path
        if not self.subscription_key or not self.endpoint:
            self.logger.error(
                "Azure credentials (AZURE_SUBSCRIPTION_KEY AND AZURE_ENDPOINT) are missing."
                )

        self.client = ComputerVisionClient(
            self.endpoint,
            CognitiveServicesCredentials(self.subscription_key)
            )

    def extract_text_from_pdf(self) -> str:
        """
        Extracts text from a local PDF file using Azure Cognitive Services' Read API.
            
        Returns:
            str: Extracted text from the PDF or error message if the extraction fails.
        """
        self.logger.info("Starting text extraction from PDF: %s", self.file_path)

        try:
            # Open the local file in binary mode
            with open(self.file_path, "rb") as file_stream:
                read_response = self.client.read_in_stream(file_stream, raw=True)

            # Get the operation location (URL with an ID at the end)
            operation_location = read_response.headers["Operation-Location"]
            operation_id = operation_location.split("/")[-1]

            # Polling the operation status
            while True:
                read_result = self.client.get_read_result(operation_id)
                if read_result.status not in ['notStarted', 'running']:
                    break
                time.sleep(1)

            # Extract text from the result if the operation was successful
            extracted_text = ""
            if read_result.status == OperationStatusCodes.succeeded:
                for page_result in read_result.analyze_result.read_results:
                    for line in page_result.lines:
                        extracted_text += "\n" + line.text
                    extracted_text += "\n\n"

            self.logger.info("Text extraction completed successfully.")
            return extracted_text if extracted_text else "No text found in the PDF."

        except AzureError as e:
            self.logger.error("Error occurred during text extraction: %s", e)
            return "Error occurred: %s" % e
