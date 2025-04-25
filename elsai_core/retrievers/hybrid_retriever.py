from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from ..config.loggerConfig import setup_logger

class HybridRetriever:
    """
    A class to perform hybrid retrieval using multiple retrievers, including BM25. 
    It combines their results using an ensemble approach and returns relevant documents.
    Attributes:
        logger: Logger instance for logging information and errors.
    """
    def __init__(self):
        """
        Initializes the HybridRetriever class and sets up logging.
        """
        self.logger = setup_logger()

    def hybrid_retrieve(self, chunks: list, retrievers: list, question: str):
        """
        Performs a hybrid retrieval using BM25 and other retrievers, and returns the results.

        Args:
            chunks (list): List of text chunks to initialize BM25 retriever.
            retrievers (list): Existing list of retrievers to be used in the ensemble.
            question (str): The query or question to perform retrieval on.

        Returns:
            list: A list of relevant documents retrieved by the ensemble of retrievers.
        """
        self.logger.info("Starting hybrid retrieval for question: '%s'.", question)

        try:
            # Initialize BM25 retriever if chunks are provided
            if chunks:
                bm25_retriever = BM25Retriever.from_texts(chunks)
                retrievers.append(bm25_retriever)
                self.logger.info("BM25 retriever initialized and added to retrievers list.")

            # Combine retrievers using an ensemble approach
            ensemble_retriever = EnsembleRetriever(
                retrievers=retrievers,
                weights=[1 / len(retrievers)] * len(retrievers) # Equal weighting for all retrievers
            )
            self.logger.info("Ensemble retriever created with dynamic retrievers.")

            # Invoke the ensemble retriever with the query
            return ensemble_retriever.invoke(question)

        except Exception as e:
            # Log the error and re-raise the exception
            self.logger.error("Error during hybrid retrieval: %s", e)
            raise RuntimeError("Hybrid retrieval failed.") from e
