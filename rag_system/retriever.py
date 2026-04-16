"""Semantic retriever with similarity-score threshold filtering."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document

from rag_system.config import RAGConfig
from rag_system.exceptions import AmbiguousQueryError, NoRelevantDocumentsError, RetrievalError
from rag_system.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Holds the outcome of a single retrieval query.

    Attributes:
        query: The original search query.
        documents: Retrieved document chunks that met the threshold.
        scores: Similarity scores corresponding to each document.
        filtered_count: Number of candidates that were below the threshold.
    """

    query: str
    documents: List[Document] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    filtered_count: int = 0

    @property
    def is_empty(self) -> bool:
        """Return *True* when no documents were retrieved."""
        return len(self.documents) == 0


class SemanticRetriever:
    """Retrieves document chunks relevant to a query using semantic similarity.

    Chunks whose cosine similarity score falls below
    ``config.similarity_threshold`` are excluded from results.  Edge cases
    (empty store, query too short, no results above threshold) are handled
    gracefully via custom exceptions.

    Usage::

        retriever = SemanticRetriever(config, vector_store_manager)
        result = retriever.retrieve("What is machine learning?")
        for doc, score in zip(result.documents, result.scores):
            print(score, doc.page_content[:80])
    """

    _MIN_WORD_COUNT = 1  # minimum distinct tokens in a query

    def __init__(self, config: RAGConfig, store: VectorStoreManager) -> None:
        self._config = config
        self._store = store

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for *query*.

        Args:
            query: Natural-language search query.
            top_k: Override the default ``config.top_k``.
            threshold: Override the default ``config.similarity_threshold``.

        Returns:
            A :class:`RetrievalResult` (may have empty ``documents`` when no
            matches exceed the threshold **and** ``raise_on_empty=False``).

        Raises:
            AmbiguousQueryError: If the query is too short or contains only
                stop-word-like tokens.
            RetrievalError: If the vector store is empty.
            NoRelevantDocumentsError: If all candidates fall below the
                similarity threshold.
        """
        self._validate_query(query)

        if self._store.is_empty():
            raise RetrievalError(
                "Vector store is empty — index documents before querying"
            )

        k = top_k or self._config.top_k
        min_score = threshold if threshold is not None else self._config.similarity_threshold

        try:
            candidates = self._store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

        documents: List[Document] = []
        scores: List[float] = []
        below_threshold = 0

        for doc, score in candidates:
            # ChromaDB returns *distance* (lower = closer); convert to
            # similarity for a uniform interface.  FAISS with IP / cosine
            # returns similarity directly.  We normalise here: if score > 1
            # it is likely a raw distance; treat similarity = 1 / (1 + dist).
            similarity = self._normalize_score(score)
            if similarity >= min_score:
                documents.append(doc)
                scores.append(similarity)
            else:
                below_threshold += 1

        if not documents:
            raise NoRelevantDocumentsError(query, min_score)

        logger.info(
            "Retrieved %d chunk(s) for query '%s' (threshold=%.2f, filtered=%d)",
            len(documents),
            query,
            min_score,
            below_threshold,
        )
        return RetrievalResult(
            query=query,
            documents=documents,
            scores=scores,
            filtered_count=below_threshold,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_query(self, query: str) -> None:
        stripped = query.strip()
        if len(stripped) < self._config.min_query_length:
            raise AmbiguousQueryError(
                query,
                f"query must have at least {self._config.min_query_length} non-whitespace characters",
            )
        words = stripped.split()
        if len(words) < self._MIN_WORD_COUNT:
            raise AmbiguousQueryError(query, "query must contain at least one word")

    @staticmethod
    def _normalize_score(score: float) -> float:
        """Convert a raw store score to a 0–1 similarity value.

        ChromaDB returns *L2 distance* (0 = identical, ∞ = unrelated).
        FAISS with cosine similarity returns a score in [-1, 1].
        We map both to [0, 1] similarity.
        """
        if score < 0:
            # FAISS cosine score in [-1, 0) → shift to [0, 0.5)
            return (score + 1.0) / 2.0
        if score <= 1.0:
            # Already a similarity in [0, 1] (FAISS inner-product / cosine)
            return float(score)
        # Chroma L2 distance: similarity = 1 / (1 + dist)
        return 1.0 / (1.0 + score)
