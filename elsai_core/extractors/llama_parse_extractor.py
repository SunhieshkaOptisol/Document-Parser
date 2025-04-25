from llama_parse import LlamaParse

class LlamaParseExtractor:
    """
    A class to interact with the LlamaParse library for loading and extracting data from files.
    """
    def __init__(self, **kwargs):
        if 'api_key' not in kwargs:
            raise ValueError("API key is required to use LlamaParse")
        self.llama_parse = LlamaParse(**kwargs)

    def load_csv(self, csv_file_path):
        """
        Loads data from a CSV file using LlamaParse.

        Args:
            csv_file_path (str): Path to the CSV file.

        Returns:
            Any: Parsed data returned by LlamaParse.
        """
        return self.llama_parse.load_data(csv_file_path)
