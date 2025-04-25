from langchain_community.document_loaders import UnstructuredExcelLoader
from elsai_core.config.loggerConfig import setup_logger

class UnstructuredExcelLoaderService:
    """
    A service for loading and extracting data from unstructured Excel files 
    using the UnstructuredExcelLoader library.
    """
    def __init__(self, file_path:str):
        self.logger = setup_logger()
        self.file_path = file_path

    def load_excel(self):
        """
        Load data from an unstructured Excel file using UnstructuredExcelLoader.

       
        Returns:
            list: Extracted documents from the Excel file.
            None: If an error occurs during loading.

        
        """
        try:
            self.logger.info("Loading %s Excel File...", self.file_path)
            loader = UnstructuredExcelLoader(self.file_path, mode="elements")
            docs = loader.load()
            self.logger.info("Loaded %d documents from the Excel file.", len(docs))
            return docs
        except FileNotFoundError:
            self.logger.error("File not found: %s", self.file_path)
            return None
        except Exception as e:
            self.logger.error("An error occurred while loading the Excel file: %s", e)
            return None
