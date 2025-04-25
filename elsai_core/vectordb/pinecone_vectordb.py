from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from elsai_core.config.loggerConfig import setup_logger
import os

class PineconeVectorDb:
    """
    PineconeVectorDb handles operations for managing and querying vectors in Pinecone.
    
    Key Methods:
    1. __init__() - Initializes Pinecone index, creating it if necessary.
    2. add_document() - Adds or updates documents in the Pinecone index.
    3. retrieve_document() - Retrieves relevant documents from the index based on embeddings and filters.
    """

    def __init__(self, index_name: str, dimension: int = 1536):
        """
        Initializes the PineconeVectorDb and ensures the index exists. If the index does not exist,
        it is created with the specified dimension.

        Args:
            index_name (str): The name of the Pinecone index to use.
            dimension (int): Dimensionality of the embeddings. Default is 1536.

        Raises:
            Exception: If Pinecone index creation or initialization fails.
        """
        self.logger = setup_logger()
        pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        existing_indexes = [index['name'] for index in pinecone.list_indexes()]
        self.index_name = index_name
        if index_name not in existing_indexes:
            self.logger.info(f"Creating Pinecone index: {self.index_name} with dimension {dimension}")
            pinecone.create_index(self.index_name, dimension=dimension,
                                  spec=ServerlessSpec(cloud='aws', region='us-east-1')
                                  )
        else:
            self.logger.info(f"Pinecone index '{self.index_name}' already exists.")

        self.index = pinecone.Index(self.index_name)
        self.logger.info(f"Pinecone index '{self.index_name}' initialized successfully.")
    
    def add_document(self, document: dict, namespace: str) -> None:
        """
        Adds a document to the Pinecone index. If the document already exists, it is updated.

        Args:
            document (dict): A dictionary containing the document ID, embeddings, and optional metadata.
            namespace (str): The namespace to add the document to.
        Raises:
            ValueError: If the document lacks required fields ('id' or 'embeddings').
            RuntimeError: If the document insertion fails for any reason.
        """
        try:
            # Ensure the document contains required fields
            if "id" not in document or "embeddings" not in document:
                raise ValueError("Document must contain 'id' and 'embeddings'.")

            # Upsert the document into Pinecone
            self.index.upsert(vectors=[(document["id"], document["embeddings"], document.get("metadatas", {}))],
                              namespace=namespace
                              )
            self.logger.info(f"Document with ID '{document['id']}' added to index '{self.index_name}'.")
            
        except Exception as exc:
            self.logger.info(f"Failed to add document to index '{self.index_name}': {exc}")
            raise RuntimeError(f"Failed to add document to Pinecone: {str(exc)}") from exc

    def retrieve_document(self, namespace:str, question_embedding: list, files_id: list, k: int = 10):
        """
        Retrieves documents from the Pinecone index by querying with embeddings and applying a file ID filter.

        Args:
            namespace (str): The namespace to query for documents.
            question_embedding (list): Embeddings to use for the similarity search.
            files_id (list): List of file IDs to filter results by.
            k (int): The number of top results to retrieve. Default is 10.

        Returns:
            dict: A dictionary of matching documents along with their metadata.

        Raises:
            Exception: If the query operation fails.
        """
        self.logger.info(f"Attempting to retrieve documents from index '{self.index_name}' with embeddings: {question_embedding} and files_id: {files_id}")
        # Construct the query filter for the file IDs
        filter_condition = {"file_id": {"$in": files_id}}
        
        # Query the Pinecone index with embeddings and filter conditions
        results = self.index.query(
            namespace=namespace,
            vector=question_embedding,
            top_k=k,
            filter=filter_condition,
            include_metadata=True
        )
        self.logger.info(f"Query successful. Retrieved {len(results['matches'])} results.")
        return results
