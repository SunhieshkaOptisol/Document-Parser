import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from elsai_core.model import AzureOpenAIConnector
from elsai_core.extractors.azure_document_intelligence import AzureDocumentIntelligence
from elsai_core.config.loggerConfig import setup_logger
from elsai_core.prompts import PezzoPromptRenderer
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

load_dotenv()
logger = setup_logger()

st.set_page_config(page_title="Document Intelligence", page_icon="📄", layout="wide")

# Initialize session state variables if they don't exist
if 'invoice_content' not in st.session_state:
    st.session_state.invoice_content = None
if 'po_content' not in st.session_state:
    st.session_state.po_content = None
if 'invoice_path' not in st.session_state:
    st.session_state.invoice_path = None
if 'po_path' not in st.session_state:
    st.session_state.po_path = None

def extract_content_from_pdf(pdf_path):
    logger.info(f"Extracting from PDF: {os.path.basename(pdf_path)}")
    try:
        doc_processor = AzureDocumentIntelligence(pdf_path)
        extracted_text = doc_processor.extract_text()
        extracted_tables = doc_processor.extract_tables()
        return extracted_text, extracted_tables
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}", exc_info=True)
        raise

def extract_content_from_pdf_direct(pdf_path):
    """
    Extract tables and text from a PDF file using Azure Document Intelligence directly.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        tuple: (extracted_text, extracted_tables)
    """
    logger.info(f"Starting extraction from PDF: {os.path.basename(pdf_path)}")
    
    try:
        # Get Azure credentials from environment variables
        endpoint = os.getenv("VISION_ENDPOINT")
        key = os.getenv("VISION_KEY")
        
        if not endpoint or not key:
            logger.error("Azure Document Intelligence credentials not found in environment variables")
            raise ValueError("Azure Document Intelligence credentials not found in environment variables")
        
        # Initialize the Document Intelligence client
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(key)
        )
        logger.debug("Document Intelligence client initialized")
        
        # Process the PDF file
        with open(pdf_path, "rb") as f:
            logger.info("Beginning document analysis")
            poller = document_intelligence_client.begin_analyze_document("prebuilt-layout", body=f)
        
        # Get the result
        logger.info("Waiting for document analysis to complete")
        result = poller.result()
        logger.info("Document analysis completed successfully")
        
        # Extract text content
        logger.debug("Extracting text content")
        extracted_text = extract_text(result)
        
        # Extract tables
        logger.debug("Extracting tables")
        extracted_tables = extract_tables(result)
        
        logger.info(f"Extraction complete. Found {len(extracted_text)} pages of text and {len(extracted_tables)} tables")
        return extracted_text, extracted_tables
    
    except Exception as e:
        logger.error(f"Error extracting content from PDF: {str(e)}", exc_info=True)
        raise

def extract_text(result):
    """
    Extract text content from the analysis result.
    
    Args:
        result: The result from Document Intelligence analysis
        
    Returns:
        dict: Dictionary containing text content by page
    """
    logger.debug("Starting text extraction from analysis result")
    text_content = {}
    
    # Extract text from paragraphs (most reliable for formatted text)
    if result.paragraphs:
        logger.debug(f"Found {len(result.paragraphs)} paragraphs to extract")
        # Sort paragraphs by their position in the document
        sorted_paragraphs = sorted(
            result.paragraphs, 
            key=lambda p: (p.spans[0].offset if p.spans else 0)
        )
        
        for paragraph in sorted_paragraphs:
            page_numbers = [region.page_number for region in paragraph.bounding_regions] if paragraph.bounding_regions else []
            
            for page_num in page_numbers:
                if page_num not in text_content:
                    text_content[page_num] = []
                
                text_content[page_num].append({
                    "type": "paragraph",
                    "content": paragraph.content,
                    "role": paragraph.role if hasattr(paragraph, "role") else None
                })
    
    # If no paragraphs, extract text from pages
    if not text_content and result.pages:
        logger.debug(f"No paragraphs found, extracting from {len(result.pages)} pages")
        for page in result.pages:
            page_num = page.page_number
            text_content[page_num] = []
            
            if page.lines:
                for line in page.lines:
                    text_content[page_num].append({
                        "type": "line",
                        "content": line.content
                    })
    
    logger.debug(f"Text extraction complete. Extracted text from {len(text_content)} pages")
    return text_content

def extract_tables(result):
    """
    Extract tables from the analysis result.
    
    Args:
        result: The result from Document Intelligence analysis
        
    Returns:
        list: List of dictionaries containing table data
    """
    logger.debug("Starting table extraction from analysis result")
    extracted_tables = []
    
    if result.tables:
        logger.debug(f"Found {len(result.tables)} tables to extract")
        for table_idx, table in enumerate(result.tables):
            logger.debug(f"Processing table {table_idx+1} with {table.row_count} rows and {table.column_count} columns")
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
            logger.debug(f"Extracted table {table_idx+1} with {len(table_data['cells'])} cells")
    
    logger.debug(f"Table extraction complete. Extracted {len(extracted_tables)} tables")
    return extracted_tables

def format_table_as_markdown(table_data):
    """
    Format extracted table data as markdown table.
    
    Args:
        table_data (dict): Table data dictionary
        
    Returns:
        str: Markdown formatted table
    """
    logger.debug(f"Formatting table {table_data.get('table_id', 'unknown')} as markdown")
    if not table_data or not table_data["cells"]:
        logger.warning("Empty table data received for markdown formatting")
        return "Empty table"
    
    # Get dimensions
    rows = table_data["row_count"]
    cols = table_data["column_count"]
    
    # Create empty grid
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    
    # Fill in the grid with cell content
    for cell in table_data["cells"]:
        row = cell["row_index"]
        col = cell["column_index"]
        grid[row][col] = cell["content"]
    
    # Convert to markdown
    markdown = []
    
    # Header row
    markdown.append("| " + " | ".join(grid[0]) + " |")
    
    # Header separator
    markdown.append("| " + " | ".join(["---" for _ in range(cols)]) + " |")
    
    # Data rows
    for row in grid[1:]:
        markdown.append("| " + " | ".join(row) + " |")
    
    logger.debug("Table markdown formatting complete")
    return "\n".join(markdown)

def format_table(tables):
    """
    Format the table into a string representation.
    
    Args:
        table: The table object to format
    Returns:
        str: Formatted table string
    """
    tables_str = ""
    tables_str += "\n\n## Tables\n"
    for i, table in enumerate(tables):
        tables_str += f"\n### Table {i+1}\n"
        tables_str += f"Pages: {', '.join(map(str, table['page_numbers']))}\n\n"
        
        # Create simple text representation of table
        rows = table["row_count"]
        cols = table["column_count"]
        grid = [["" for _ in range(cols)] for _ in range(rows)]
        
        for cell in table["cells"]:
            row = cell["row_index"]
            col = cell["column_index"]
            grid[row][col] = cell["content"]
        
        for row in grid:
            tables_str += " | ".join(row) + "\n"
    return tables_str

def convert_to_markdown(text_content, tables):
    """
    Convert extracted text and tables to a single markdown string.
    
    Args:
        text_content (dict): Extracted text content by page
        tables (list): Extracted tables
        
    Returns:
        str: Combined markdown formatted string
    """
    logger.debug("Converting extracted content to markdown")
    markdown_parts = []
    
    # Add document title
    markdown_parts.append("# Extracted PDF Content\n")
    
    # Add text content
    markdown_parts.append("## Text Content\n")
    for page_num in sorted(text_content.keys()):
        markdown_parts.append(f"### Page {page_num}\n")
        for item in text_content[page_num]:
            markdown_parts.append(item["content"])
            markdown_parts.append("\n")
        markdown_parts.append("\n")
    
    # Add tables
    if tables:
        markdown_parts.append("## Tables\n")
        for i, table in enumerate(tables):
            markdown_parts.append(f"### Table {i+1}\n")
            markdown_parts.append(f"*Pages: {', '.join(map(str, table['page_numbers']))}*\n\n")
            markdown_parts.append(format_table_as_markdown(table))
            markdown_parts.append("\n\n")
    
    logger.debug("Markdown conversion complete")
    return "".join(markdown_parts)

def extract_content_from_csv(csv_path):
    logger.info(f"Extracting from CSV: {os.path.basename(csv_path)}")
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        # Convert dataframe to text for processing
        text_content = df.to_string(index=False)
        # Tables are the actual dataframe representation
        tables = [df]
        return text_content, tables
    except Exception as e:
        logger.error(f"CSV extraction error: {str(e)}", exc_info=True)
        raise

def process_file(file_path, document_type):
    try:
        # Determine file type based on extension
        if file_path.lower().endswith('.pdf'):
            content = extract_content_from_pdf(file_path)
        elif file_path.lower().endswith('.csv'):
            content = extract_content_from_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {os.path.splitext(file_path)[1]}")
        
        # Store in session state
        if document_type == "invoice":
            st.session_state.invoice_content = content
        else:
            st.session_state.po_content = content
            
        return content
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

def get_prompt_name_by_type(document_type):
    """
    Get the appropriate Pezzo prompt name based on document type.
    
    Args:
        document_type (str): The type of document
        
    Returns:
        str: Pezzo prompt name
    """
    document_type_lower = document_type.lower()
    
    if "invoice" in document_type_lower and "timesheet" in document_type_lower:
        return "CombinedParserPrompt"
    elif "multiple" in document_type_lower and "timesheet" in document_type_lower:
        return "MultipleTimesheetParserPrompt"
    elif "invoice" in document_type_lower:
        return "InvoiceParserPromptt"
    elif "timesheet" in document_type_lower:
        return "TimesheetParserPrompt"
    else:
        logger.warning(f"Unknown document type: {document_type}, defaulting to InvoiceParserPromptt")
        return "InvoiceParserPromptt"

def process_pdf(uploaded_file, document_type):
    """
    Process an uploaded PDF file for advanced document parsing.
    
    Args:
        uploaded_file: The uploaded file object from Streamlit
        document_type: The type of document ('invoice', 'timesheet', etc.)
        
    Returns:
        str: Markdown formatted results
    """
    file_name = uploaded_file.name
    logger.info(f"Processing PDF file: {file_name} as {document_type}")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
        logger.debug(f"Created temporary file: {tmp_path}")
    
    try:
        # Extract content from PDF
        logger.info("Extracting content from PDF")
        text_content, tables = extract_content_from_pdf_direct(tmp_path)
        
        # Convert to markdown
        logger.info("Converting extracted content to markdown")
        markdown_content = convert_to_markdown(text_content, tables)
        logger.debug("Markdown conversion completed")
        
        # Initialize Azure OpenAI connector
        logger.info("Initializing LLM connector")
        connector = AzureOpenAIConnector()
        llm = connector.connect_azure_open_ai(deploymentname="gpt-4o-mini")
        logger.info("LLM connector initialized")
        
        # Initialize Pezzo prompt renderer
        logger.info("Initializing Pezzo prompt renderer")
        renderer = PezzoPromptRenderer(
            api_key=st.secrets["PEZZO_API_KEY"],
            project_id=st.secrets["PEZZO_PROJECT_ID_2"],
            environment=st.secrets["PEZZO_ENVIRONMENT"],
            server_url=st.secrets["PEZZO_SERVER_URL"]
        )
        logger.info("Pezzo prompt renderer initialized")
        
        # Get appropriate prompt name based on document type
        prompt_name = get_prompt_name_by_type(document_type)
        logger.info(f"Using Pezzo prompt: {prompt_name}")
        
        # Get prompt from Pezzo
        prompt = renderer.get_prompt(prompt_name)
        logger.debug(f"Retrieved prompt from Pezzo: {prompt_name}")
        
        # Format the prompt with document content
        prompt_txt = f"{prompt}\n\nDocument Content: {markdown_content}"
        
        # Send to LLM
        logger.info("Sending request to LLM")
        response = llm.invoke(prompt_txt)
        result = response.content
        logger.info(f"Received response from LLM ({len(result)} characters)")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing PDF {file_name}: {str(e)}", exc_info=True)
        return f"Error processing PDF: {str(e)}"
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            logger.debug(f"Cleaning up temporary file: {tmp_path}")
            os.unlink(tmp_path)
            logger.debug("Temporary file removed")
    
def generate_comparison_summary(invoice_path, po_path):
    # Check if content exists in session state
    if st.session_state.invoice_content is None or st.session_state.po_content is None:
        # Try to extract content if paths are available but content is not
        if st.session_state.invoice_path and st.session_state.invoice_content is None:
            process_file(st.session_state.invoice_path, "invoice")
        if st.session_state.po_path and st.session_state.po_content is None:
            process_file(st.session_state.po_path, "purchase_order")
            
        # Check again after trying to extract
        if st.session_state.invoice_content is None or st.session_state.po_content is None:
            return "Error: Invoice or PO content not available. Please extract both documents first."

    connector = AzureOpenAIConnector()
    llm = connector.connect_azure_open_ai(deploymentname="gpt-4o-mini")
    renderer = PezzoPromptRenderer(
        api_key=st.secrets["PEZZO_API_KEY"],
        project_id=st.secrets["PEZZO_PROJECT_ID"],
        environment=st.secrets["PEZZO_ENVIRONMENT"],
        server_url=st.secrets["PEZZO_SERVER_URL"]
    )
 
    prompt_name = "PurchaseOrder"
    prompt = renderer.get_prompt(prompt_name)
 
    prompt_txt = f"{prompt}\n\nText: {st.session_state.invoice_content},{st.session_state.po_content}"
    response = llm.invoke(prompt_txt)
    return response.content

def process_invoice_pdf(file_path):
    """
    Process a single invoice PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        str: LLM processed results
    """
    file_name = os.path.basename(file_path)
    logger.info(f"Processing PDF file: {file_name}")
    
    try:
        # Extract content from PDF
        text_content, tables = extract_content_from_pdf(file_path)
        tables_str = format_table(tables)
        
        # Process with LLM
        connector = AzureOpenAIConnector()
        llm = connector.connect_azure_open_ai(deploymentname="gpt-4o-mini")
           
        # Generate prompt for LLM
        renderer = PezzoPromptRenderer(
            api_key=st.secrets["PEZZO_API_KEY"],
            project_id=st.secrets["PEZZO_PROJECT_ID_1"],
            environment=st.secrets["PEZZO_ENVIRONMENT"],
            server_url=st.secrets["PEZZO_SERVER_URL"]
        )
        prompt = renderer.get_prompt("InvoiceParsingPrompt") 
        prompt_txt = prompt + f"""The content is as follows: Text from the document : {text_content} , Tables from the document : {tables_str}"""
        
        response = llm.invoke(prompt_txt)
        result = response.content
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing PDF {file_name}: {str(e)}", exc_info=True)
        return f"Error processing PDF: {str(e)}"

def process_uploaded_file(uploaded_file, document_type):
    st.write(f"Processing {document_type}: {uploaded_file.name}")
    progress = st.progress(0)
    status = st.empty()
    status.text("Saving file...")
    
    # Get file extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    # Create temp file with appropriate extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name
 
    try:
        progress.progress(20)
        status.text("Extracting data...")
        result = process_file(temp_path, document_type)
   
        progress.progress(100)
        status.text("Done ✅")
 
        with st.expander(f"Results: {uploaded_file.name}", expanded=True):
            st.markdown("Extracted Successfully")
        
        # Store the path in session state
        if document_type == "invoice":
            st.session_state.invoice_path = temp_path
        else:
            st.session_state.po_path = temp_path
            
        return result, temp_path
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, temp_path

def invoice_parser_app():
    """
    Invoice Parser Application
    """
    st.title("Invoice Parser")
    st.markdown("Upload invoice PDFs to extract structured data")
    
    # Check for environment variables
    vision_endpoint = st.secrets.get("VISION_ENDPOINT")
    vision_key = st.secrets.get("VISION_KEY")
    
    if not vision_endpoint or not vision_key:
        st.error("Azure Document Intelligence credentials not found. Please check your environment variables.")
        return
    
    # File uploader section
    st.subheader("Upload PDF Invoices")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True, key="invoice_parser_uploader")
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} file(s)")
        
        # Process files when user clicks the button
        if st.button("Process Files"):
            for uploaded_file in uploaded_files:
                # Create progress bar for this file
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text(f"Processing: {uploaded_file.name}...")
                
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                
                try:
                    progress_bar.progress(25)
                    status_text.text(f"Analyzing document: {uploaded_file.name}")
                    
                    # Process the PDF
                    result = process_invoice_pdf(temp_file_path).replace("```markdown","").replace("```","")
                    
                    progress_bar.progress(100)
                    status_text.text(f"Completed: {uploaded_file.name}")
                    
                    # Display results in an expander
                    with st.expander(f"Results for {uploaded_file.name}", expanded=True):
                        st.markdown(result)
                        
                        # Add download button for the results
                        st.download_button(
                            label="Download results as markdown",
                            data=result,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_results.md",
                            mime="text/markdown"
                        )               
                except Exception as e:
                    progress_bar.progress(100)
                    status_text.text(f"Error processing: {uploaded_file.name}")
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                
            st.success("All files processed!")
    else:
        st.info("Please upload PDF invoice files to begin")

def invoice_po_comparison_app():
    """
    Invoice and PO Comparison Application
    """
    st.title("📄 Invoice and PO Comparison")
    st.markdown("Upload Invoices or Purchase Orders (PDF or CSV) and compare them.")
 
    # Check for environment variables
    vision_endpoint = st.secrets.get("VISION_ENDPOINT")
    vision_key = st.secrets.get("VISION_KEY")
    if not vision_endpoint or not vision_key:
        st.error("Azure Vision credentials missing.")
        return
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("Invoice")
        invoice_file = st.file_uploader("Upload Invoice (PDF or CSV)", type=["pdf", "csv"], key="invoice_comparison")
        if invoice_file and st.button("Extract Invoice"):
            _, temp_path = process_uploaded_file(invoice_file, "invoice")
 
    with col2:
        st.subheader("Purchase Order")
        po_file = st.file_uploader("Upload PO (PDF or CSV)", type=["pdf", "csv"], key="po_comparison")
        if po_file and st.button("Extract Purchase Order"):
            _, temp_path = process_uploaded_file(po_file, "purchase_order")
 
    st.markdown("---")
    st.subheader("📊 Compare Invoice and Purchase Order")
 
    # Debug information
    if st.checkbox("Show debug info"):
        st.write("Invoice path:", st.session_state.invoice_path)
        st.write("PO path:", st.session_state.po_path)
        st.write("Invoice content exists:", st.session_state.invoice_content is not None)
        st.write("PO content exists:", st.session_state.po_content is not None)
 
    if st.session_state.invoice_path and st.session_state.po_path:
        if st.button("Compare Documents"):
            with st.spinner("Generating comparison summary..."):
                comparison = generate_comparison_summary(st.session_state.invoice_path, st.session_state.po_path)
                st.markdown(comparison)
                st.download_button(
                    "Download Comparison",
                    data=comparison,
                    file_name="invoice_po_comparison.md",
                    mime="text/markdown"
                )
    else:
        st.info("Upload both Invoice and Purchase Order to enable comparison.")

def document_parsing_app():
    """
    Document Parsing Application
    """
    st.title("Document Parser")
    st.markdown("Upload PDFs of invoices, timesheets, combined documents or multiple documents for detailed extraction")
    
    # Check for environment variables
    vision_endpoint = st.secrets.get("VISION_ENDPOINT")
    vision_key = st.secrets.get("VISION_KEY")
    
    if not vision_endpoint or not vision_key:
        st.error("Azure Document Intelligence credentials not found. Please check your environment variables.")
        return
    
    # Document type selection dropdown
    document_type = st.selectbox(
        "Select document type",
        options=["Invoice", "Timesheet", "Digital Invoice and Timesheet", "Multiple Timesheets"],
        help="Select the type of document you are uploading"
    )
    logger.info(f"Document type selected: {document_type}")
    
    # File uploader
    uploaded_files = st.file_uploader("Upload PDF documents", 
                                     type=['pdf'], 
                                     accept_multiple_files=True,
                                     key="advanced_parser_uploader")
    
    if uploaded_files:
        logger.info(f"{len(uploaded_files)} files uploaded")
        
        if st.button("Process Files", key="process_advanced_files"):
            logger.info("Process Files button clicked")
            
            with st.spinner("Processing files..."):
                # Process each file
                for uploaded_file in uploaded_files:
                    logger.info(f"Processing file: {uploaded_file.name}")
                    st.subheader(f"Processing: {uploaded_file.name}")
                    
                    # Process the file with the selected document type
                    result = process_pdf(uploaded_file, document_type).replace("```markdown","").replace("```","")
                    
                    # Create a container for the rendered markdown
                    table_container = st.container()
                    with table_container:
                        # Render the markdown as a table
                        st.markdown(result, unsafe_allow_html=True)
                    
                    # Add download button
                    st.download_button(
                        label="Download results as markdown",
                        data=result,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_results.md",
                        mime="text/markdown"
                    )
                    
                    # Add a divider between files
                    st.markdown("---")
                    logger.info(f"Completed processing file: {uploaded_file.name}")
                
                logger.info("All files processed successfully")
                st.success("All files processed successfully!")
    else:
        st.info("Please upload PDF files to begin")

def main():
    """
    Main Streamlit application with app selector
    """
    # Create a sidebar with app selection
    st.sidebar.title("Document Intelligence")
    app_mode = st.sidebar.selectbox(
        "Choose Application Mode",
        ["Invoice Parser", "Invoice-PO Comparison", "Document Parser (AU)"]
    )
    
    # Display app information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        This application provides three tools:
        
        - **Invoice Parser**: Process individual and multilingual invoices to extract key information.
        - **Invoice-PO Comparison**: Compare invoices against purchase orders to identify discrepancies.
        - **Document Parser (AU)**: Extract structured information from invoices, timesheets, combined documents or multiple documents.
        """
    )
    
    # Run the selected app
    if app_mode == "Invoice Parser":
        invoice_parser_app()
    elif app_mode == "Invoice-PO Comparison":
        invoice_po_comparison_app()
    else:
        document_parsing_app()

if __name__ == "__main__":
    try:
        logger.info("Streamlit application starting")
        main()
        logger.info("Streamlit application session ended")
    except Exception as e:
        logger.critical(f"Unhandled exception in Streamlit application: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")