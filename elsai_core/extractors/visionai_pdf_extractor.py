import os
import base64
from io import BytesIO
from openai import OpenAI
from pdf2image import convert_from_path
from langchain_core.documents import Document
from elsai_core.config.loggerConfig import setup_logger




class VisionAIExtractor:
    """
    VisionAIPDFExtractor is a class that interacts with OpenAI Vision AI client
    to extract text from PDFs.
    """
    def __init__(self, api_key, model_name="gpt-4o"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.logger = setup_logger()

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a given PDF page using the 
        Vision AI client and returns as Langchain Documents.

        Args:
            pdf_path: The path to the PDF file.

        Returns:
            str: List of Langchain Documents containing the extracted text from each page.
        """
        images = convert_from_path(pdf_path)
        documents = []
        for page_num, page_image in enumerate(images, start=1):
            documents.append(self.__get_image_as_document(page_num, page_image, pdf_path))
        return documents

    def __get_image_as_document(self, page_num, page_image, file_path):
        buffer = BytesIO()
        try :
            page_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that responds in Markdown.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Convert the following PDF page to markdown. "
                                "Return only the markdown with no explanation text. "
                                "Do not exclude any content from the page. "
                                "Do not include delimiters like '''markdown or '''.\n\n"
                                "Replace images with brief [descriptive summaries], "
                                "and use appropriate markdown syntax (headers [#, ##, ###, ####], bold **, italic *). "
                                "Output should be clean, formatted markdown that matches the original layout.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],
                temperature=0.0,
            )
            page_content = response.choices[0].message.content.strip()
            return Document(
                page_content=page_content,
                metadata={
                    "page_num": page_num,
                    "source": os.path.basename(file_path),
                }
            )
        except Exception as e:
            self.logger.error("Error while processing page %d of %s: %s", page_num, file_path, e)
            raise e
        finally:
            buffer.close()
