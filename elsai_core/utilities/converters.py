"""
This module contains utility functions for converting documents.
"""

from langchain_core.documents import Document
from elsai_core.config.loggerConfig import setup_logger

class DocumentConverter:
    """
    A class to convert documents from llama index format to langchain format.
    """
    
    def __init__(self):
        self.logger = setup_logger()
    
    def llama_index_to_langchain_document(self, llama_index_document, file_name=""):
        """
        Convert a llama index document to a langchain document.

        :param llama_index_document: The document in llama index format.
        :param file_name: The name of the file to be used as metadata.
        :return: A langchain document.
        """
        langchain_document = Document(
            page_content=llama_index_document.text_resource.text,
            metadata= {
                "source": file_name
            }
        )

        return langchain_document