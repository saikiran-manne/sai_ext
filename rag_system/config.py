"""Configuration dataclass for the RAG system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RAGConfig:
    """Central configuration for the RAG pipeline.

    Attributes:
        chunk_size: Number of characters per document chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
        similarity_threshold: Minimum cosine-similarity score for a retrieved
            chunk to be included in results (0.0 – 1.0).
        top_k: Maximum number of chunks to retrieve per query.
        embedding_model: HuggingFace model name used for sentence embeddings.
        collection_name: Name of the ChromaDB collection.
        persist_directory: Optional path to persist the ChromaDB collection on
            disk.  When *None* an ephemeral in-memory store is used.
        min_query_length: Minimum number of non-whitespace characters required
            before a query is dispatched to the retriever.
    """

    chunk_size: int = 500
    chunk_overlap: int = 100
    similarity_threshold: float = 0.3
    top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_name: str = "rag_documents"
    persist_directory: Optional[str] = None
    min_query_length: int = 3

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.min_query_length <= 0:
            raise ValueError("min_query_length must be positive")
