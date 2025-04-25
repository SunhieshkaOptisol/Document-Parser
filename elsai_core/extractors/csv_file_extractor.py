
from elsai_core.config.loggerConfig import setup_logger
from langchain_community.document_loaders.csv_loader import CSVLoader
class CSVFileExtractor:
    def __init__(self, file_path:str):
        self.logger = setup_logger()
        self.file_path = file_path

    def load_from_csv(self):
        """
        Load data from a CSV file using CSVLoader.

      

        Returns:
            list: Extracted data from the CSV file.

        Raises:
            Exception: If the CSV file cannot be loaded.
        """
        try:
            self.logger.info("Attempting to load CSV file: %s", self.file_path)
            loader = CSVLoader(self.file_path)
            extracted_data = loader.load()
            self.logger.info("Successfully loaded data from CSV file: %s", self.file_path)
            return extracted_data
        except Exception as e:
            self.logger.error("Failed to load CSV file: %s. Error: %s", self.file_path, str(e))
            raise
