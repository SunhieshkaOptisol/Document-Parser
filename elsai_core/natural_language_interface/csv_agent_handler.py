from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from elsai_core.config.loggerConfig import setup_logger
class CSVAgentHandler:
    """
    Handles interaction with a CSV-based AI agent.
    """
    def __init__(self, csv_files, model="None", verbose=True, agent_type:str=None):

        """
        Initialize the CSV agent.
        :param csv_files: Path to one or more CSV files (string or list).
        :param model: Instance of a chat model (e.g., ChatOpenAI or AzureChatOpenAI).
        :param verbose: Show detailed logs if True.
        :param agent_type: Type of agent to use. Defaults to OPENAI_FUNCTIONS if not specified.
        """
        self.csv_files = csv_files
        self.model = model
        self.verbose = verbose
        self.logger = setup_logger()
        self.logger.info("Initializing CSV agent.")

        if self.model is None:
            self.logger.error("No model provided. Please provide a model.")
            raise ValueError("No model provided. Please provide a model.")
        if agent_type is None:
            agent_type = AgentType.OPENAI_FUNCTIONS
            self.logger.info("No agent_type provided. Defaulting to OPENAI_FUNCTIONS.")
        self.agent = create_csv_agent(
            self.model,
            self.csv_files,
            verbose=self.verbose,
            agent_type=agent_type,
            allow_dangerous_code=True
        )
        self.logger.info("CSV agent created successfully.")


    def ask_question(self, query):
        """
        Ask a question to the CSV agent.
        :param query: Question to ask about the CSV data.
        :return: Answer from the agent.
        """
        self.logger.info("Asking question: %s", query)
        if not self.agent:
            self.logger.error("Agent not created. Call create_agent() first.")
            raise RuntimeError("Agent not created. Call create_agent() first.")
        response = self.agent.run(query)
        self.logger.info("Response received: %s", response)
        return response
    