"""
This module provides utilities for splitting documents into chunks.
"""
import re
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from elsai_core.config.loggerConfig import setup_logger

class DocumentChunker:
    """
    Class for chunking content into pages and returning as Document objects with metadata.
    """

    def __init__(self):
        self.logger = setup_logger()

    def chunk_page_wise(self, contents: str, file_name: str) -> list:
        """
        Splits the contents into chunks when there are two or more consecutive
        newline characters and returns each section as a Document object with
        page numbers and filename in the metadata

        Args:
            contents (str): Input text
            file_name (str): Name of the file

        Returns:
            list: list of text chunks
        """
        self.logger.debug("Starting to chunk the document")
        pages = re.split(r'\n\n+', contents)
        document_pages = []
        for index, page in enumerate(pages):
            self.logger.debug("Processing page %d", index + 1)
            document = Document(page_content=page, metadata={"page_number": index + 1, "source": file_name})
            document_pages.append(document)

        self.logger.info("Document chunking completed. %d pages processed.", len(document_pages))
        return document_pages

    def chunk_markdown_header_wise(
        self,
        text: str = "",
        file_name: str = "",
        headers_to_split_on: list[tuple[str, str]] = None,
        strip_headers: bool = True,
    ) -> list[str]:
        """
        Split the text into markdown headers.
        """
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=strip_headers)
        split_text = markdown_splitter.split_text(text)
        for item in split_text:
            item.metadata["source"] = file_name
        return split_text
