"""Custom exceptions for the RAG system."""


class RAGException(Exception):
    """Base exception for all RAG system errors."""


class DocumentLoadError(RAGException):
    """Raised when a document cannot be loaded."""

    def __init__(self, path: str, reason: str = "") -> None:
        self.path = path
        self.reason = reason
        message = f"Failed to load document '{path}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class VectorStoreError(RAGException):
    """Raised when a vector store operation fails."""


class EmbeddingError(RAGException):
    """Raised when an embedding operation fails."""


class RetrievalError(RAGException):
    """Raised when a retrieval operation fails."""


class NoRelevantDocumentsError(RetrievalError):
    """Raised when no documents meet the similarity threshold for a query."""

    def __init__(self, query: str, threshold: float) -> None:
        self.query = query
        self.threshold = threshold
        super().__init__(
            f"No documents found above similarity threshold {threshold} for query: '{query}'"
        )


class AmbiguousQueryError(RetrievalError):
    """Raised when the query is too ambiguous to retrieve meaningful results."""

    def __init__(self, query: str, reason: str = "") -> None:
        self.query = query
        self.reason = reason
        message = f"Query is ambiguous: '{query}'"
        if reason:
            message += f" — {reason}"
        super().__init__(message)
