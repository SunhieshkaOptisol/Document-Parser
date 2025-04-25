import chromadb
import os
from elsai_core.config.loggerConfig import setup_logger

class ChromaVectorDb:
    def __init__(self, chroma_host: str = None, chroma_port: int = 8000):
        """
        Initializes the ChromaVectorDb class with the specified ChromaDB host and port.

        Args:
            chroma_host (str, optional): The host for the ChromaDB instance. Defaults to None.
            chroma_port (int, optional): The port for the ChromaDB instance. Defaults to 8000.
        """
        if chroma_host is None:
            chroma_host = os.getenv('CHROMA_HOST')
        self.chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        self.chroma_client.api_version = "v1"
        self.logger = setup_logger()

    def create_if_not_exists(self, collection_name: str):
        """
        Checks if a collection exists in ChromaDB, and creates it if it does not exist.

        Args:
            collection_name (str): The name of the collection to check or create.
        """
        self.logger.info(f"Checking if collection '{collection_name}' exists.")
        existing_collections = self.chroma_client.list_collections()
        if not any(collection.name == collection_name for collection in existing_collections):
            self.logger.info(f"Collection '{collection_name}' not found, creating it.")
            self.chroma_client.create_collection(name=collection_name)

    def add_document(self, document, collection_name: str) -> None:
        """
        Adds a document to a ChromaDB collection.

        Args:
            document (dict): A dictionary containing 'id', 'embeddings', 'page_content', and 'metadatas'.
            collection_name (str): The name of the collection to add the document to.
        """
        self.logger.info(f"Adding document with id '{document['id']}' to collection '{collection_name}'.")
        try:
            self.create_if_not_exists(collection_name)
            collection = self.chroma_client.get_collection(name=collection_name)
        except Exception as exc:
            self.logger.error(f"Error retrieving or creating collection: {exc}")
            raise RuntimeError(f"collection:'{collection_name}' does not exist") from exc
        collection.add(ids=[document["id"]],
                       embeddings=[document["embeddings"]],
                       documents=[document["page_content"]],
                       metadatas=[document['metadatas']])
        self.logger.info(f"Document with id '{document['id']}' added successfully.")

    def retrieve_document(self, collection_name: str, embeddings: list, files_id: list=None, k: int = 10):
        """
        Retrieves documents from a ChromaDB collection based on the query embeddings and file IDs.

        Args:
            collection_name (str): The name of the collection to query.
            embeddings (list): The embeddings to use for the query.
            files_id (list): The list of file IDs to filter by.
            k (int, optional): The number of results to retrieve. Defaults to 10.

        Returns:
            dict: The results of the query.
        """
        self.logger.info(f"Retrieving documents from collection '{collection_name}' with k={k}.")
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
        except Exception as exc:
            self.logger.error(f"Error retrieving collection: {exc}")
            raise RuntimeError(f"collection:'{collection_name}' does not exist") from exc
        query_filter = {}
        if files_id:
            query_filter = {"file_id": {"$in": files_id}}
        results = collection.query(
            query_embeddings=[embeddings],
            n_results=k,
            where=query_filter
        )
        self.logger.info(f"Retrieved {len(results['documents'])} documents.")
        return results

    def get_collection(self, collection_name):
        """
        Retrieves a ChromaDB collection by name.

        Args:
            collection_name (str): The name of the collection to retrieve.

        Returns:
            Collection: The ChromaDB collection.
        """
        self.logger.info(f"Retrieving collection '{collection_name}'.")
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            return collection
        except Exception as exc:
            self.logger.error(f"Error retrieving collection: {exc}")
            raise RuntimeError(f"collection:'{collection_name}' does not exist") from exc

    def fetch_chunks(self, collection_name: str, files_id: list):
        """
        Fetches text chunks from the ChromaDB collection based on the collection name.

        Args:
            collection_name (str): The name of the collection to fetch data from.

        Returns:
            list: A list of text chunks from the collection.
        """
        self.logger.info(f"Fetching chunks from collection '{collection_name}'.")
        collection = self.chroma_client.get_collection(name=collection_name)
        collection_results = collection.get(
            where={"file_id": {"$in": files_id}}
        )
        chunks = [item for item in collection_results['documents'] if item]
        self.logger.info(f"Fetched {len(chunks)} chunks.")
        return chunks
    
    def delete_collection(self, collection_name:str):
        """
        Deletes a collection from ChromaDB.

        Args:
            collection_name (str): The name of the collection to delete.
        """
        collection_exists = any(col.name == collection_name for col in self.chroma_client.list_collections())
        if collection_exists:
            self.logger.info(f"Deleting collection '{collection_name}'.")
            try:
                self.chroma_client.delete_collection(name=collection_name)
            except Exception as exc:
                self.logger.error(f"Error deleting collection: {exc}")
                raise RuntimeError(f"collection:'{collection_name}' does not exist") from exc
        else:
            self.logger.error(f"Collection '{collection_name}' does not exist.")
            raise RuntimeError(f"Collection '{collection_name}' does not exist.")