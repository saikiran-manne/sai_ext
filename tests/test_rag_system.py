"""Tests for the RAG system components.

These tests run entirely in-process using in-memory structures so they are
fast and require no external services or API keys.
"""

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag_system.config import RAGConfig
from rag_system.document_loader import DocumentLoader
from rag_system.evaluator import RetrievalEvaluator, EvaluationResult
from rag_system.exceptions import (
    AmbiguousQueryError,
    DocumentLoadError,
    NoRelevantDocumentsError,
    RAGException,
    RetrievalError,
    VectorStoreError,
)
from rag_system.rag_pipeline import RAGPipeline
from rag_system.retriever import RetrievalResult, SemanticRetriever
from rag_system.text_splitter import DocumentChunker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(content: str, source: str = "test.txt") -> Document:
    return Document(page_content=content, metadata={"source": source})


def _make_retrieval_result(
    query: str, texts: List[str], scores: List[float]
) -> RetrievalResult:
    docs = [_make_doc(t) for t in texts]
    return RetrievalResult(query=query, documents=docs, scores=scores)


# ---------------------------------------------------------------------------
# RAGConfig tests
# ---------------------------------------------------------------------------


class TestRAGConfig:
    def test_default_values(self):
        cfg = RAGConfig()
        assert cfg.chunk_size == 500
        assert cfg.chunk_overlap == 100
        assert cfg.similarity_threshold == 0.3
        assert cfg.top_k == 5

    def test_custom_values(self):
        cfg = RAGConfig(chunk_size=200, chunk_overlap=50)
        assert cfg.chunk_size == 200
        assert cfg.chunk_overlap == 50

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            RAGConfig(chunk_size=0)

    def test_invalid_chunk_overlap_negative(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            RAGConfig(chunk_overlap=-1)

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            RAGConfig(chunk_size=100, chunk_overlap=100)

    def test_invalid_similarity_threshold(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            RAGConfig(similarity_threshold=1.5)

    def test_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            RAGConfig(top_k=0)


# ---------------------------------------------------------------------------
# DocumentLoader tests
# ---------------------------------------------------------------------------


class TestDocumentLoader:
    def test_load_txt_file(self, tmp_path):
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("Hello, this is a test document.", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load(txt_file)

        assert len(docs) == 1
        assert "Hello" in docs[0].page_content

    def test_load_missing_file_raises(self, tmp_path):
        loader = DocumentLoader()
        with pytest.raises(DocumentLoadError, match="does not exist"):
            loader.load(tmp_path / "missing.txt")

    def test_load_unsupported_extension_raises(self, tmp_path):
        doc_file = tmp_path / "file.docx"
        doc_file.write_bytes(b"fake content")

        loader = DocumentLoader()
        with pytest.raises(DocumentLoadError, match="unsupported file type"):
            loader.load(doc_file)

    def test_load_directory_single_txt(self, tmp_path):
        (tmp_path / "a.txt").write_text("Document A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Document B", encoding="utf-8")
        (tmp_path / "ignore.md").write_text("Markdown", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path)

        assert len(docs) == 2  # only .txt files
        contents = {d.page_content for d in docs}
        assert "Document A" in contents
        assert "Document B" in contents

    def test_load_directory_missing_raises(self, tmp_path):
        loader = DocumentLoader()
        with pytest.raises(DocumentLoadError, match="does not exist"):
            loader.load_directory(tmp_path / "no_such_dir")

    def test_load_directory_not_a_directory_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("content")
        loader = DocumentLoader()
        with pytest.raises(DocumentLoadError, match="not a directory"):
            loader.load_directory(f)

    def test_load_directory_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.txt").write_text("Root doc", encoding="utf-8")
        (sub / "child.txt").write_text("Child doc", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path, recursive=True)

        assert len(docs) == 2


# ---------------------------------------------------------------------------
# DocumentChunker tests
# ---------------------------------------------------------------------------


class TestDocumentChunker:
    def test_split_produces_chunks(self):
        config = RAGConfig(chunk_size=50, chunk_overlap=10)
        chunker = DocumentChunker(config)
        doc = _make_doc("word " * 100)  # ~500 chars

        chunks = chunker.split([doc])

        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        config = RAGConfig(chunk_size=100, chunk_overlap=20)
        chunker = DocumentChunker(config)
        doc = _make_doc("a" * 500)

        chunks = chunker.split([doc])

        for chunk in chunks:
            assert len(chunk.page_content) <= 100 + 20  # small tolerance

    def test_overlap_creates_shared_content(self):
        config = RAGConfig(chunk_size=50, chunk_overlap=20)
        chunker = DocumentChunker(config)
        text = "abcdefghij" * 20  # 200 chars
        doc = _make_doc(text)

        chunks = chunker.split([doc])
        # The end of one chunk should appear at the start of the next
        if len(chunks) >= 2:
            tail = chunks[0].page_content[-10:]
            head = chunks[1].page_content[:30]
            assert tail in head or head in chunks[0].page_content

    def test_empty_documents_returns_empty(self):
        config = RAGConfig()
        chunker = DocumentChunker(config)
        assert chunker.split([]) == []


# ---------------------------------------------------------------------------
# VectorStoreManager tests (use FAISS for speed; no ChromaDB daemon needed)
# ---------------------------------------------------------------------------


class TestVectorStoreManager:
    """Uses a small sentence-transformers model via mocked embeddings to keep
    tests fast without downloading large models."""

    def _mock_embeddings(self):
        """Return a mock Embeddings object producing deterministic 8-dim vectors."""
        import numpy as np
        from langchain_core.embeddings import Embeddings

        class _FakeEmbeddings(Embeddings):
            _dim = 8

            def embed_documents(self, texts):
                rng = np.random.default_rng(0)
                return [rng.random(self._dim).tolist() for _ in texts]

            def embed_query(self, text):
                rng = np.random.default_rng(0)
                return rng.random(self._dim).tolist()

        return _FakeEmbeddings()

    def test_add_and_search_faiss(self):
        from rag_system.vector_store import VectorStoreManager

        config = RAGConfig()
        embeddings = self._mock_embeddings()

        manager = VectorStoreManager(config, embeddings, backend="faiss")
        docs = [_make_doc("The sky is blue"), _make_doc("Cats like fish")]
        manager.add_documents(docs)

        assert not manager.is_empty()
        results = manager.similarity_search_with_score("sky", k=2)
        assert len(results) == 2

    def test_empty_store_raises_on_search(self):
        from rag_system.vector_store import VectorStoreManager

        config = RAGConfig()
        embeddings = self._mock_embeddings()
        manager = VectorStoreManager(config, embeddings, backend="faiss")

        with pytest.raises(VectorStoreError, match="empty"):
            manager.similarity_search_with_score("query")

    def test_clear_resets_store(self):
        from rag_system.vector_store import VectorStoreManager

        config = RAGConfig()
        embeddings = self._mock_embeddings()
        manager = VectorStoreManager(config, embeddings, backend="faiss")
        manager.add_documents([_make_doc("content")])
        manager.clear()
        assert manager.is_empty()

    def test_unsupported_backend_raises(self):
        from rag_system.vector_store import VectorStoreManager

        with pytest.raises(VectorStoreError, match="Unsupported backend"):
            VectorStoreManager(RAGConfig(), MagicMock(), backend="pinecone")


# ---------------------------------------------------------------------------
# SemanticRetriever tests
# ---------------------------------------------------------------------------


class TestSemanticRetriever:
    def _mock_store(self, results):
        """Return a mock VectorStoreManager yielding *results*."""
        store = MagicMock()
        store.is_empty.return_value = False
        store.similarity_search_with_score.return_value = results
        return store

    def test_retrieve_returns_above_threshold(self):
        config = RAGConfig(similarity_threshold=0.5, top_k=5)
        doc_a = _make_doc("Machine learning is a subset of AI")
        doc_b = _make_doc("Irrelevant content about cooking")
        # score <= 1 → treated as similarity directly
        store = self._mock_store([(doc_a, 0.9), (doc_b, 0.1)])

        retriever = SemanticRetriever(config, store)
        result = retriever.retrieve("What is machine learning?")

        assert len(result.documents) == 1
        assert result.documents[0] == doc_a
        assert result.filtered_count == 1

    def test_no_results_above_threshold_raises(self):
        config = RAGConfig(similarity_threshold=0.9, top_k=5)
        store = self._mock_store([(_make_doc("unrelated"), 0.1)])

        retriever = SemanticRetriever(config, store)
        with pytest.raises(NoRelevantDocumentsError):
            retriever.retrieve("something specific")

    def test_empty_store_raises(self):
        config = RAGConfig()
        store = MagicMock()
        store.is_empty.return_value = True

        retriever = SemanticRetriever(config, store)
        with pytest.raises(RetrievalError, match="empty"):
            retriever.retrieve("query")

    def test_short_query_raises_ambiguous(self):
        config = RAGConfig(min_query_length=5)
        store = MagicMock()
        store.is_empty.return_value = False

        retriever = SemanticRetriever(config, store)
        with pytest.raises(AmbiguousQueryError):
            retriever.retrieve("ab")  # too short

    def test_score_normalization_chroma_distance(self):
        """L2 distance > 1 should be normalised to (0, 0.5)."""
        config = RAGConfig(similarity_threshold=0.0, top_k=5)
        doc = _make_doc("content")
        store = self._mock_store([(doc, 2.0)])  # L2 distance

        retriever = SemanticRetriever(config, store)
        result = retriever.retrieve("query")
        assert result.scores[0] == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_retrieve_result_is_empty_property(self):
        result = RetrievalResult(query="test", documents=[], scores=[])
        assert result.is_empty

        result2 = _make_retrieval_result("test", ["doc"], [0.8])
        assert not result2.is_empty


# ---------------------------------------------------------------------------
# RetrievalEvaluator tests
# ---------------------------------------------------------------------------


class TestRetrievalEvaluator:
    def test_perfect_precision_and_recall(self):
        evaluator = RetrievalEvaluator()
        result = _make_retrieval_result(
            "machine learning",
            ["Machine learning is great", "Deep learning is a subset of ML"],
            [0.9, 0.85],
        )
        eval_result = evaluator.evaluate(
            result,
            relevant_snippets=["machine learning", "deep learning"],
        )
        assert eval_result.precision == pytest.approx(1.0)
        assert eval_result.recall == pytest.approx(1.0)
        assert eval_result.f1 == pytest.approx(1.0)

    def test_zero_precision_irrelevant_docs(self):
        evaluator = RetrievalEvaluator()
        result = _make_retrieval_result(
            "query",
            ["cats and dogs", "pizza and pasta"],
            [0.8, 0.7],
        )
        eval_result = evaluator.evaluate(result, relevant_snippets=["machine learning"])
        assert eval_result.precision == 0.0
        assert eval_result.recall == 0.0
        assert eval_result.f1 == 0.0

    def test_partial_recall(self):
        evaluator = RetrievalEvaluator()
        result = _make_retrieval_result(
            "query",
            ["machine learning is interesting"],
            [0.9],
        )
        eval_result = evaluator.evaluate(
            result,
            relevant_snippets=["machine learning", "neural network"],
        )
        # 1 retrieved, 1 relevant → precision=1.0; 1/2 relevant found → recall=0.5
        assert eval_result.precision == pytest.approx(1.0)
        assert eval_result.recall == pytest.approx(0.5)

    def test_empty_snippets_returns_zeros(self):
        evaluator = RetrievalEvaluator()
        result = _make_retrieval_result("query", ["some doc"], [0.8])
        eval_result = evaluator.evaluate(result, relevant_snippets=[])
        assert eval_result.precision == 0.0
        assert eval_result.recall == 0.0
        assert eval_result.total_relevant_count == 0

    def test_batch_evaluate(self):
        evaluator = RetrievalEvaluator()
        r1 = _make_retrieval_result("q1", ["relevant content here"], [0.9])
        r2 = _make_retrieval_result("q2", ["irrelevant stuff"], [0.8])

        results = evaluator.batch_evaluate(
            [r1, r2],
            [["relevant content"], ["machine learning"]],
        )
        assert len(results) == 2
        assert results[0].precision == 1.0
        assert results[1].precision == 0.0

    def test_batch_evaluate_length_mismatch_raises(self):
        evaluator = RetrievalEvaluator()
        with pytest.raises(ValueError, match="same length"):
            evaluator.batch_evaluate(
                [_make_retrieval_result("q", ["doc"], [0.5])],
                [["a"], ["b"]],
            )

    def test_aggregate_metrics(self):
        evaluator = RetrievalEvaluator()
        r1 = _make_retrieval_result("q1", ["relevant content here"], [0.9])
        r2 = _make_retrieval_result("q2", ["relevant text here"], [0.85])

        eval_results = evaluator.batch_evaluate(
            [r1, r2],
            [["relevant content"], ["relevant text"]],
        )
        agg = evaluator.aggregate_metrics(eval_results)
        assert "macro_precision" in agg
        assert agg["macro_precision"] == pytest.approx(1.0)
        assert agg["macro_recall"] == pytest.approx(1.0)

    def test_aggregate_metrics_empty(self):
        evaluator = RetrievalEvaluator()
        agg = evaluator.aggregate_metrics([])
        assert agg == {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}


# ---------------------------------------------------------------------------
# RAGPipeline integration tests (using mocks to avoid model downloads)
# ---------------------------------------------------------------------------


class TestRAGPipelineEdgeCases:
    """Test edge cases in the RAGPipeline without a real embedding model."""

    def test_query_before_ingest_raises(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._config = RAGConfig()
        pipeline._retriever = None
        pipeline._store = None
        pipeline._evaluator = RetrievalEvaluator()

        with pytest.raises(RAGException, match="ingest"):
            pipeline.query("something")

    def test_pipeline_is_ready_false_before_ingest(self, tmp_path):
        """Pipeline reports not ready before documents are loaded."""
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._config = RAGConfig()
        pipeline._store = None
        pipeline._retriever = None

        assert not pipeline.is_ready

    def test_ingest_txt_with_mock_embeddings(self, tmp_path):
        """Full ingest flow with a mocked embedding model."""
        txt = tmp_path / "test.txt"
        txt.write_text(
            "Retrieval-Augmented Generation (RAG) combines retrieval with generation. "
            "It fetches relevant documents and uses them to answer questions accurately. "
            "LangChain provides tools for building RAG systems with various vector stores.",
            encoding="utf-8",
        )

        import numpy as np
        from unittest.mock import patch

        mock_embed = MagicMock()
        mock_embed.embed_documents.side_effect = lambda texts: [
            np.random.default_rng(42).random(384).tolist() for _ in texts
        ]
        mock_embed.embed_query.side_effect = lambda text: (
            np.random.default_rng(42).random(384).tolist()
        )

        with patch(
            "rag_system.embeddings.EmbeddingsFactory.get_embeddings",
            return_value=mock_embed,
        ):
            pipeline = RAGPipeline(backend="faiss")
            count = pipeline.ingest(txt)

        assert count >= 1
        assert pipeline.is_ready

    def test_reset_clears_pipeline(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._config = RAGConfig()
        mock_store = MagicMock()
        pipeline._store = mock_store
        pipeline._retriever = MagicMock()
        pipeline._evaluator = RetrievalEvaluator()

        pipeline.reset()

        assert pipeline._store is None
        assert pipeline._retriever is None

    def test_evaluate_delegates_to_evaluator(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._evaluator = RetrievalEvaluator()

        result = _make_retrieval_result("query", ["machine learning content"], [0.9])
        eval_result = pipeline.evaluate(result, relevant_snippets=["machine learning"])

        assert eval_result.precision == 1.0


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_document_load_error_message(self):
        exc = DocumentLoadError("file.pdf", "not found")
        assert "file.pdf" in str(exc)
        assert "not found" in str(exc)

    def test_no_relevant_docs_error_message(self):
        exc = NoRelevantDocumentsError("my query", 0.5)
        assert "my query" in str(exc)
        assert "0.5" in str(exc)

    def test_ambiguous_query_error(self):
        from rag_system.exceptions import AmbiguousQueryError

        exc = AmbiguousQueryError("x", "too short")
        assert "x" in str(exc)
        assert "too short" in str(exc)

    def test_rag_exception_hierarchy(self):
        assert issubclass(DocumentLoadError, RAGException)
        assert issubclass(VectorStoreError, RAGException)
        assert issubclass(RetrievalError, RAGException)
        assert issubclass(NoRelevantDocumentsError, RetrievalError)
        assert issubclass(AmbiguousQueryError, RetrievalError)
