"""Embeddings factory supporting HuggingFace sentence-transformers."""

import logging
from functools import lru_cache
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from rag_system.config import RAGConfig
from rag_system.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingsFactory:
    """Creates and caches embedding model instances.

    Using a local HuggingFace model avoids the need for external API keys
    while still producing high-quality semantic embeddings.

    Usage::

        factory = EmbeddingsFactory(config)
        embeddings = factory.get_embeddings()
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Return a (cached) embedding model instance.

        Returns:
            A :class:`~langchain_huggingface.HuggingFaceEmbeddings` instance
            ready to embed texts.

        Raises:
            EmbeddingError: If the model cannot be loaded.
        """
        if self._embeddings is None:
            self._embeddings = self._build_embeddings(self._config.embedding_model)
        return self._embeddings

    @staticmethod
    @lru_cache(maxsize=4)
    def _build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
        try:
            logger.info("Loading embedding model '%s'", model_name)
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{model_name}': {exc}"
            ) from exc
