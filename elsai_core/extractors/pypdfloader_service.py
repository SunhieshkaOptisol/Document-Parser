from langchain_community.document_loaders import PyPDFLoader
from elsai_core.config.loggerConfig import setup_logger

class PyPDFTextExtractor:
    """
    A class to extract text content from PDF files using the PyPDFLoader library.
    This class handles the initialization of a logger, 
    loading the PDF file, and extracting its text content.
    """
    def __init__(self, file_path:str):
        self.logger = setup_logger()
        self.file_path = file_path

    def extract_text_from_pdf(self)->str:
        """
            Extracts text from the PDF file.

            Returns:
                str: The extracted text or an error message.
        """
        try:
            self.logger.info("Starting PDF extraction from %s", self.file_path)
            loader = PyPDFLoader(self.file_path)
            docs = loader.load()
            extracted_contents = docs[0].page_content
            return extracted_contents if extracted_contents else "No text contents found in the PDF"

        except FileNotFoundError as e:
            self.logger.error("File not found: %s", e)
            return f"Error occurred: {e}"

        except Exception as e:
            self.logger.error("Error while extracting text from %s: %s", self.file_path, e)
            return f"Error occurred: {e}"
