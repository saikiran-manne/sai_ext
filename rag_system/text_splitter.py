"""Document chunking with configurable size and overlap."""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_system.config import RAGConfig

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Splits ``Document`` objects into overlapping chunks using
    :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`.

    The default parameters match the project specification:
    ``chunk_size=500``, ``chunk_overlap=100``.

    Usage::

        chunker = DocumentChunker(config)
        chunks = chunker.split(documents)
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents into chunks.

        Args:
            documents: Raw ``Document`` objects (e.g. from
                :class:`~rag_system.document_loader.DocumentLoader`).

        Returns:
            A list of ``Document`` chunks, each with ``start_index`` injected
            into its metadata by the underlying splitter.
        """
        if not documents:
            logger.warning("split() called with an empty document list")
            return []

        chunks = self._splitter.split_documents(documents)
        logger.info(
            "Split %d document(s) into %d chunk(s) "
            "(chunk_size=%d, chunk_overlap=%d)",
            len(documents),
            len(chunks),
            self._config.chunk_size,
            self._config.chunk_overlap,
        )
        return chunks
