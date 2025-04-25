import os
from langchain_core.prompts import ChatPromptTemplate 
from elsai_core.config.loggerConfig import setup_logger
from elsai_core.prompts import PezzoPromptRenderer


class SummarizationService:

    """
        A class for generation summaries using an LLM
    """
    def __init__(self, llm):
        self.logger = setup_logger()
        self.llm = llm
    
    def summarize(self, text:str)->str:
        """
            Generates a summary for the given text.

            text-> Input text to summarize

            return: the summarized output 
        """
        prompt_renderer = PezzoPromptRenderer(
        api_key=os.getenv("PEZZO_API_KEY"),
        project_id=os.getenv("PEZZO_PROJECT_ID"),
        environment=os.getenv("PEZZO_ENVIRONMENT"),
        server_url=os.getenv("PEZZO_SERVER_URL"),
    )
        prompt = prompt_renderer.get_prompt("SummarizationPrompt")
        prompt_template = ChatPromptTemplate.from_template(prompt)
        prompt_input = prompt_template.format_messages(text=text)

        # prompt = (
        #     "You are an expert summarization assistant. Your task is to generate a clear, concise, and informative summary.\n"
        #     "### Instructions:\n"
        #     "1. Capture the main idea and key points.\n"
        #     "2. Keep it brief but meaningful.\n"
        #     "3. Use simple and precise language.\n\n"
        #     "### Text to Summarize:\n"
        #     f"{text}\n\n"
        #     "### Summary:"
        # )
        try:
            response = self.llm(
                prompt_input
            )
            if hasattr(response, "content"):
                return response.content.strip()
            elif isinstance(response, str):  # Handling plain string response
                return response.strip()

        except Exception as e:
            self.logger.error(f"Error during LLM summarization: {str(e)}")
            return "Error: Could not generate a summary at this time."



    

