from elsai_core.config.loggerConfig import setup_logger
from docling.document_converter import DocumentConverter

class DoclingPDFTextExtractor:
    """
    A class to extract text from PDF files using AWS Textract after uploading to S3.
    It handles authentication, file upload, text extraction, and cleanup in AWS S3.
    """

    def __init__(self, file_path:str):
        self.logger = setup_logger()
        self.file_path = file_path

    def extract_text_from_pdf(self) -> str:
        """
        Extracts text from a PDF file using docling.

        

        Returns:
            str: Extracted text content from the PDF file.
        """
        try:
            self.logger.info("Starting PDF extraction from %s", self.file_path)
            converter = DocumentConverter()
            result = converter.convert(self.file_path)
            extracted_text = result.document.export_to_markdown()
            return extracted_text

        except FileNotFoundError as e:
            self.logger.error("File not found: %s: %s", self.file_path, e)
            return "File not found: %s" % e
        except Exception as e:
            self.logger.error("Error while extracting text from %s: %s", self.file_path, e)
            return "Error occurred: %s" % e
