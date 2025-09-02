import json
import os
from dotenv import load_dotenv
import requests
import chromadb
from chromadb.types import Where
from models.unixcoder import UniXcoder
from reranker import Result

import torch
import logging

from typing import Optional, List

# Initialize logger
logger = logging.getLogger(__name__)

load_dotenv()

HUGGINGFACE_URL = os.getenv("HUGGINGFACE_URL")

class ChromaDBQueryEngine:
    def __init__(self, collection_name, persist_directory, limit=5):
        """
        Initialize the ChromaDB query engine.

        Args:
            collection_name (str): The name of the collection to query.
            persist_directory (str): Directory where the ChromaDB database is stored.
            limit (int): Maximum number of results to return.
        """
        logger.info(
            "Initializing ChromaDBQueryEngine with collection: %s, persist directory: %s",
            collection_name,
            persist_directory,
        )
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_collection(name=collection_name)

        self.limit = limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UniXcoder("microsoft/unixcoder-base")
        logger.info("UniXcoder model loaded and moved to device: %s", self.device)
        self.model.to(self.device)

    def embedd_query(self, query: str) -> List[float]:
        tokens_id = self.model.tokenize(
            [f"{query}"], max_length=512, mode="<encoder-only>"
        )
        source_id = torch.tensor(tokens_id, device=self.device)
        _, embedding = self.model(source_id)
        embedding = embedding.tolist()
        logger.info("Embedding generated successfully. Length: %d", len(embedding))
        return embedding

    def rerank_hits(self, query: str, result: Result) -> Result:
        # construct payload
        payload = {
            "query": query,
            "hits": [(i + 1, item["content"]) for i, item in enumerate(result.hits)]
        }

        assert HUGGINGFACE_URL is not None, "HUGGINGFACE_URL is not set"

        response = requests.post(HUGGINGFACE_URL, json=payload)
        data = json.loads(response.text)

        assert data["query"] == query
        hits = data["reranked_hits"]

        result = Result(
            query=query,
            hits=[{"content": item[1]} for item in hits],
        )
        return result


    def execute_query_nl(self, query: str, where: Optional[Where] = None) -> Result:
        """
        Execute a natural language query and retrieve results with optional filters.

        Args:
            query (str): The natural language query.
            where (Where, optional): A filter expression supporting 'and', 'or', and key-value pairs.

        Returns:
            Result: A result object containing ranked results with metadata.

        Example:
            query = "What is the purpose of the module?"
            where = {"$or": [{"type": "class"}, {"type": "function"}]}
            results = engine.execute_query_nl(query, where=where)

            # For AND filtering:
            where = {"$and": [{"type": "class"}, {"author": "Alice"}]}
        """
        logger.info("Executing natural language query: %s", query)
        embedding = self.embedd_query(query)

        results = self.collection.query(
            query_embeddings=embedding, n_results=self.limit, where=where
        )

        documents = results["documents"] or [[]]
        metadatas = results["metadatas"] or [[]]

        # Todo implement logging for preliminary chromadb query results
        logger.info(
            "Query executed successfully. Number of results: %d",
            len(documents),
        )

        # sanitize documents, empty docs should be replaced by the name of the function/class
        sanitized_documents = []
        for doc, meta in zip(documents[0], metadatas[0]):
            if not doc:
                doc = meta["name"]
            sanitized_documents.append(doc)

        result = Result(
            query=query,
            hits=[{"content": content} for content in sanitized_documents]
        )
        result = self.rerank_hits(query, result)
        return result

    def execute_query_e(self, embedding: list[float]):
        pass

engine = ChromaDBQueryEngine("manim_embeddings", "../data/chromadb/")
