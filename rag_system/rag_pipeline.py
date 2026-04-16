"""High-level RAG pipeline orchestrating all components."""

import logging
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.documents import Document

from rag_system.config import RAGConfig
from rag_system.document_loader import DocumentLoader
from rag_system.embeddings import EmbeddingsFactory
from rag_system.evaluator import EvaluationResult, RetrievalEvaluator
from rag_system.exceptions import RAGException
from rag_system.retriever import RetrievalResult, SemanticRetriever
from rag_system.text_splitter import DocumentChunker
from rag_system.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    Orchestrates document loading, chunking, indexing, and retrieval in a
    single cohesive API.

    Usage::

        pipeline = RAGPipeline()                    # default config
        pipeline.ingest("docs/")                    # load & index directory
        result = pipeline.query("What is RAG?")     # retrieve relevant chunks

        # Evaluate retrieval quality
        eval_result = pipeline.evaluate(result, relevant_snippets=["retrieval-augmented"])

    Args:
        config: Optional :class:`~rag_system.config.RAGConfig` instance.
            A default config is used when not provided.
        backend: Vector store backend — ``"chroma"`` (default) or ``"faiss"``.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        backend: str = "chroma",
    ) -> None:
        self._config = config or RAGConfig()
        self._loader = DocumentLoader()
        self._chunker = DocumentChunker(self._config)
        self._embeddings_factory = EmbeddingsFactory(self._config)
        self._store: Optional[VectorStoreManager] = None
        self._retriever: Optional[SemanticRetriever] = None
        self._evaluator = RetrievalEvaluator()
        self._backend = backend

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        source: Union[str, Path, List[Document]],
        recursive: bool = False,
    ) -> int:
        """Load, chunk, and index documents.

        Args:
            source: Either a file path, a directory path, or a pre-built list
                of :class:`~langchain_core.documents.Document` objects.
            recursive: When *source* is a directory and this is *True*,
                subdirectories are searched recursively.

        Returns:
            The number of chunks indexed.

        Raises:
            RAGException: If loading or indexing fails.
        """
        try:
            documents = self._load(source, recursive)
            chunks = self._chunker.split(documents)
            self._index(chunks)
            return len(chunks)
        except RAGException:
            raise
        except Exception as exc:
            raise RAGException(f"Ingestion failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """Retrieve relevant document chunks for a natural-language *query*.

        Args:
            query: The search query.
            top_k: Override the default ``config.top_k``.
            threshold: Override the default ``config.similarity_threshold``.

        Returns:
            A :class:`~rag_system.retriever.RetrievalResult`.

        Raises:
            RAGException: If the pipeline has not been populated with documents,
                the query is ambiguous, or no relevant documents are found.
        """
        if self._retriever is None:
            raise RAGException(
                "Pipeline is not initialised — call ingest() before query()"
            )
        return self._retriever.retrieve(query, top_k=top_k, threshold=threshold)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        retrieval_result: RetrievalResult,
        relevant_snippets: List[str],
    ) -> EvaluationResult:
        """Compute precision / recall / F1 for a retrieval result.

        Args:
            retrieval_result: Outcome of a previous :meth:`query` call.
            relevant_snippets: Ground-truth text snippets that a relevant
                chunk should contain.

        Returns:
            An :class:`~rag_system.evaluator.EvaluationResult`.
        """
        return self._evaluator.evaluate(retrieval_result, relevant_snippets)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all indexed documents and reset the pipeline."""
        if self._store is not None:
            self._store.clear()
        self._store = None
        self._retriever = None
        logger.info("RAGPipeline reset")

    @property
    def config(self) -> RAGConfig:
        """Return the active configuration."""
        return self._config

    @property
    def is_ready(self) -> bool:
        """Return *True* when documents have been ingested and the pipeline is
        ready to answer queries."""
        return self._store is not None and not self._store.is_empty()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(
        self,
        source: Union[str, Path, List[Document]],
        recursive: bool,
    ) -> List[Document]:
        if isinstance(source, list):
            return source
        path = Path(source)
        if path.is_dir():
            return self._loader.load_directory(path, recursive=recursive)
        return self._loader.load(path)

    def _index(self, chunks: List[Document]) -> None:
        embeddings = self._embeddings_factory.get_embeddings()
        if self._store is None:
            self._store = VectorStoreManager(self._config, embeddings, self._backend)
            self._retriever = SemanticRetriever(self._config, self._store)
        self._store.add_documents(chunks)
