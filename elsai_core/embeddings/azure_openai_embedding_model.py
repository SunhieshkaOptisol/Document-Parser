import os
from langchain_openai import AzureOpenAIEmbeddings
from elsai_core.config.loggerConfig import setup_logger

class AzureOpenAIEmbeddingModel:
    """
    Class for embedding text and documents using Azure OpenAI Embeddings API.
    """
    def __init__(
            self,
            model: str = 'text-embedding-ada-002',
            azure_deployment: str = None,
            azure_endpoint: str = None,
            azure_api_key: str = None,
            azure_api_version: str = None
        ):
        self.logger = setup_logger()
        if azure_deployment is None:
            azure_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_api_key is None:
            azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if azure_api_version is None:
            azure_api_version = os.getenv("OPENAI_API_VERSION")
        self.azure_embeddings_model = AzureOpenAIEmbeddings(
            model=model,
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            openai_api_version=azure_api_version
        )

    def embed_query(self, text: str) -> list:
        """Embeds the given text using Azure OpenAI's embed_query method,
          returning the embedding vector.
        """
        try:
            self.logger.info("Starting embedding process.")
            embedding = self.azure_embeddings_model.embed_query(text)
            self.logger.info("Embedding generated successfully.")
            return embedding
        except Exception as e:
            self.logger.error("Embedding generation failed: %s", e)
            return []

    def embed_documents(self, texts: list) -> list:
        """Embeds the given list of texts using Azure OpenAI's embed_documents method,
          returning the embedding vectors.
        """
        try:
            self.logger.info("Starting embedding process.")
            embedding = self.azure_embeddings_model.embed_documents(texts)
            self.logger.info("Embedding generated successfully.")
            return embedding
        except Exception as e:
            self.logger.error("Embedding generation failed: %s",e)
            return []

    def get_embedding_model(self):
        """Returns the Azure OpenAI embedding model."""
        return self.azure_embeddings_model
