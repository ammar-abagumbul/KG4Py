import chromadb
from chromadb.config import Settings
from chromadb import Documents

from unixcoder import UniXcoder
import torch


class ChromaDBQueryEngine:
    def __init__(self, collection_name, persist_directory, limit=5):
        """
        Initialize the ChromaDB query engine.

        Args:
            collection_name (str): The name of the collection to query.
            persist_directory (str): Directory where the ChromaDB database is stored.
            limit (int): Maximum number of results to return.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.chroma_client = chromadb.PersistentClient(
            path="../data/chromadb/",
        )
        self.collection = self.chroma_client.get_collection(name=collection_name)

        self.limit = limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UniXcoder("microsoft/unixcoder-base")
        self.model.to(self.device)

    def embedd_query(self, query: str) -> list[float]:
        tokens_id = self.model.tokenize(
            [f"{query}"], max_length=512, mode="<encoder-only>"
        )
        source_id = torch.tensor(tokens_id, device=self.device)
        _, embedding = self.model(source_id)
        embedding = embedding.tolist()
        return embedding

    def execute_query_nl(self, query: str) -> list[list[Documents]]:
        embedding = self.embedd_query(query)
        result = self.collection.query(query_embeddings=embedding, n_results=self.limit)
        return result.documents

    def execute_query_e(self, embedding: list[float]):
        pass


engine = ChromaDBQueryEngine("manim_embeddings", "../data/chromadb/")
