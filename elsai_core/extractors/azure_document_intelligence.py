import os
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from elsai_core.config.loggerConfig import setup_logger
class AzureDocumentIntelligence:
    """
    Class to handle document analysis using Azure Document Intelligence.
    """

    def __init__(self, file_path:str):
        self.logger = setup_logger()
        # Set up API key and endpoint
        self.key = os.environ["VISION_KEY"]
        self.endpoint = os.environ["VISION_ENDPOINT"]
        self.file_path = file_path
        # Initialize the Document Intelligence Client
        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )

    def extract_text(self, pages: str = None) -> str:
        """
        Extracts text from a document with optional page selection.

        Args:
            
            pages (str, optional): Specific pages to analyze (e.g., "1,3"). Defaults to None.

        Returns:
            str: Extracted text content from the document.
        """

        self.logger.info("Starting text extraction from %s", self.file_path)
        try:

            with open(self.file_path, "rb") as f:
                self.logger.info("Opened file: %s", self.file_path)
                poller = self.client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    body=f,
                    content_type="application/octet-stream",
                    pages=pages
                )

            self.logger.info("Analysis started for %s. Waiting for result...", self.file_path)
            # Get the result of the analysis
            result = poller.result()
            self.logger.info("Analysis completed for %s", self.file_path)
            ocr_output = result.as_dict()
            self.logger.info("Text extraction from %s completed successfully.", self.file_path)
            return ocr_output['content']

        except Exception as e:
            self.logger.error("Error while extracting text from %s: %s", self.file_path, e)
            raise
    def extract_tables(self, pages: str = None):
        """
        Extracts tables from a document with optional page selection.
        
        Args:
            pages (str, optional): Specific pages to analyze (e.g., "1,3"). Defaults to None.
            
        Returns:
            list: List of dictionaries containing table data.
        """
        self.logger.info("Starting table extraction from %s", self.file_path)
        try:
            with open(self.file_path, "rb") as f:
                self.logger.info("Opened file: %s", self.file_path)
                poller = self.client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    body=f,
                    content_type="application/octet-stream",
                    pages=pages
                )
                
            self.logger.info("Analysis started for %s. Waiting for result...", self.file_path)
            # Get the result of the analysis
            result = poller.result()
            self.logger.info("Analysis completed for %s", self.file_path)
            
            extracted_tables = []
            
            if result.tables:
                self.logger.info(f"Found {len(result.tables)} tables to extract")
                for table_idx, table in enumerate(result.tables):
                    self.logger.info(f"Processing table {table_idx+1} with {table.row_count} rows and {table.column_count} columns")
                    # Create a table representation
                    table_data = {
                        "table_id": table_idx,
                        "row_count": table.row_count,
                        "column_count": table.column_count,
                        "page_numbers": [],
                        "cells": []
                    }
                    
                    # Add page numbers where this table appears
                    if table.bounding_regions:
                        for region in table.bounding_regions:
                            if region.page_number not in table_data["page_numbers"]:
                                table_data["page_numbers"].append(region.page_number)
                    
                    # Extract cell data
                    for cell in table.cells:
                        cell_data = {
                            "row_index": cell.row_index,
                            "column_index": cell.column_index,
                            "content": cell.content,
                            "is_header": cell.kind == "columnHeader" if hasattr(cell, "kind") else False,
                            "spans": cell.column_span if hasattr(cell, "column_span") else 1
                        }
                        table_data["cells"].append(cell_data)
                    
                    extracted_tables.append(table_data)
                    self.logger.info(f"Extracted table {table_idx+1} with {len(table_data['cells'])} cells")
            
            self.logger.info(f"Table extraction complete. Extracted {len(extracted_tables)} tables")
            return extracted_tables
                
        except Exception as e:
            self.logger.error("Error while extracting tables from %s: %s", self.file_path, e)
            raise

