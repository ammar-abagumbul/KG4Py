import chromadb
from chromadb import QueryResult
from transformers.generation.utils import SampleDecoderOnlyOutput
from models.unixcoder import UniXcoder
from reranker import Result, RankListwiseOSLLM

import torch
import logging

from typing import Optional, List, Dict

# Initialize logger
logger = logging.getLogger(__name__)


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

        self.chroma_client = chromadb.PersistentClient(
            path="../data/chromadb/",
        )
        self.collection = self.chroma_client.get_collection(name=collection_name)

        self.limit = limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UniXcoder("microsoft/unixcoder-base")
        self.reranker =
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

    def rerank_hits(self, hits) -> Result:
        ...


    def execute_query_nl(self, query: str, or_filters: Optional[List[Dict[str, str]]] = None) -> Result:
        """
        Execute a natural language query and retrieve results with optional filters.

        Args:
            query (str): The natural language query.
            filters (dict): Optional filters to apply to the results.

        Returns:
            list[dict]: A list of ranked results with metadata.

        Example:
            query = "What is the purpose of the module?"
            filters = {"type": "class"}
            results = engine.execute_query_nl(query, filters)

        future work: more complex filters, e.g., regex matching, multiple conditions
        """
        logger.info("Executing natural language query: %s", query)
        embedding = self.embedd_query(query)

        if or_filters is not None and len(or_filters) > 1:
            where = {"$or": or_filters}
        elif or_filters is not None and len(or_filters) == 1:
            # If only one filter is provided, use it directly
            where = or_filters[0]
        else:
            where = None

        results = self.collection.query(
            query_embeddings=embedding, n_results=self.limit, where=where
        )
        documents = results["documents"] or []
        logger.info(
            "Query executed successfully. Number of results: %d",
            len(documents),
        )
        result = Result(
            query=query,
            hits=[{"content": content} for content in documents]
        )
        result = self.rerank_hits(result)
        return result

    def execute_query_e(self, embedding: list[float]):
        pass

    def rerank_results(self, results: List[dict], query: str) -> List[dict]:
        """
        Rerank the results based on the query.

        Args:
            results (list[dict]): The initial results to rerank.
            query (str): The original query used for reranking.

        Returns:
            list[dict]: The reranked results.
        """
        # Placeholder for reranking logic
        logger.warning("Reranking logic is not implemented yet.")
        raise NotImplementedError("Reranking logic is not implemented yet.")


engine = ChromaDBQueryEngine("manim_embeddings", "../data/chromadb/")
