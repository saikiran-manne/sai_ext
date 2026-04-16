"""Vector store management backed by ChromaDB (or FAISS as a fallback)."""

import logging
from typing import List, Optional

from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag_system.config import RAGConfig
from rag_system.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages a ChromaDB (default) or FAISS vector store.

    Usage::

        manager = VectorStoreManager(config, embeddings)
        manager.add_documents(chunks)
        results = manager.similarity_search_with_score("query", k=5)
    """

    def __init__(
        self,
        config: RAGConfig,
        embeddings: HuggingFaceEmbeddings,
        backend: str = "chroma",
    ) -> None:
        """
        Args:
            config: RAG pipeline configuration.
            embeddings: Embedding model used to vectorise text.
            backend: ``"chroma"`` (default) or ``"faiss"``.
        """
        if backend not in {"chroma", "faiss"}:
            raise VectorStoreError(f"Unsupported backend '{backend}'; choose 'chroma' or 'faiss'")

        self._config = config
        self._embeddings = embeddings
        self._backend = backend
        self._store: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store, creating it if necessary.

        Args:
            documents: Document chunks to index.

        Raises:
            VectorStoreError: On any indexing failure.
        """
        if not documents:
            logger.warning("add_documents() called with empty list — nothing indexed")
            return

        try:
            if self._store is None:
                self._store = self._create_store(documents)
            else:
                self._store.add_documents(documents)

            logger.info(
                "Indexed %d chunk(s) into %s vector store",
                len(documents),
                self._backend,
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to index documents: {exc}") from exc

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[tuple]:
        """Perform a similarity search and return (Document, score) pairs.

        Args:
            query: The search query.
            k: Number of results to return; defaults to ``config.top_k``.

        Returns:
            A list of ``(Document, score)`` tuples sorted by descending
            similarity score.

        Raises:
            VectorStoreError: If the store is empty or the search fails.
        """
        if self._store is None:
            raise VectorStoreError(
                "Vector store is empty — call add_documents() first"
            )

        k = k or self._config.top_k
        try:
            return self._store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

    def is_empty(self) -> bool:
        """Return *True* when no documents have been indexed yet."""
        return self._store is None

    def clear(self) -> None:
        """Drop all indexed documents."""
        self._store = None
        logger.info("Vector store cleared")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_store(self, documents: List[Document]) -> object:
        if self._backend == "chroma":
            kwargs = {
                "documents": documents,
                "embedding": self._embeddings,
                "collection_name": self._config.collection_name,
            }
            if self._config.persist_directory:
                kwargs["persist_directory"] = self._config.persist_directory
            return Chroma.from_documents(**kwargs)

        # FAISS backend
        return FAISS.from_documents(documents, self._embeddings)
