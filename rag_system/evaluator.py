"""Retrieval evaluation metrics: precision and recall."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Set

from rag_system.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Stores the precision / recall metrics for a single query evaluation.

    Attributes:
        query: The query that was evaluated.
        precision: Fraction of retrieved documents that are relevant.
        recall: Fraction of all relevant documents that were retrieved.
        f1: Harmonic mean of precision and recall.
        retrieved_count: Total number of retrieved documents.
        relevant_retrieved_count: Number of retrieved documents that are
            relevant (true positives).
        total_relevant_count: Total number of ground-truth relevant documents.
    """

    query: str
    precision: float
    recall: float
    f1: float
    retrieved_count: int
    relevant_retrieved_count: int
    total_relevant_count: int

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(query={self.query!r}, "
            f"precision={self.precision:.3f}, "
            f"recall={self.recall:.3f}, "
            f"f1={self.f1:.3f})"
        )


class RetrievalEvaluator:
    """Computes precision, recall, and F1 for retrieval results.

    Relevance is determined by checking whether any ground-truth *relevant
    snippet* is contained in a retrieved chunk's text (case-insensitive
    substring match).  This is a lightweight heuristic suitable for unit
    tests and offline evaluation without a labelled dataset.

    For production use, replace :meth:`_is_relevant` with a domain-specific
    labelling function (e.g. BM25 overlap, exact-match IDs, or human labels).

    Usage::

        evaluator = RetrievalEvaluator()
        result = evaluator.evaluate(
            retrieval_result,
            relevant_snippets=["relevant phrase", "another key sentence"],
        )
        print(result.precision, result.recall)
    """

    def evaluate(
        self,
        retrieval_result: RetrievalResult,
        relevant_snippets: List[str],
    ) -> EvaluationResult:
        """Evaluate a :class:`~rag_system.retriever.RetrievalResult`.

        Args:
            retrieval_result: The outcome of a retrieval query.
            relevant_snippets: A list of text strings that a *relevant*
                document chunk should contain at least one of.

        Returns:
            An :class:`EvaluationResult` with precision, recall and F1.
        """
        if not relevant_snippets:
            logger.warning(
                "evaluate() called with no relevant_snippets — "
                "precision and recall are undefined; returning zeros"
            )
            return EvaluationResult(
                query=retrieval_result.query,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                retrieved_count=len(retrieval_result.documents),
                relevant_retrieved_count=0,
                total_relevant_count=0,
            )

        normalized_snippets: List[str] = [s.lower() for s in relevant_snippets]
        retrieved = retrieval_result.documents
        retrieved_count = len(retrieved)

        relevant_retrieved = sum(
            1
            for doc in retrieved
            if self._is_relevant(doc.page_content, normalized_snippets)
        )

        total_relevant = len(relevant_snippets)

        precision = relevant_retrieved / retrieved_count if retrieved_count else 0.0
        recall = relevant_retrieved / total_relevant if total_relevant else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        logger.info(
            "Evaluation for '%s': precision=%.3f recall=%.3f F1=%.3f",
            retrieval_result.query,
            precision,
            recall,
            f1,
        )
        return EvaluationResult(
            query=retrieval_result.query,
            precision=precision,
            recall=recall,
            f1=f1,
            retrieved_count=retrieved_count,
            relevant_retrieved_count=relevant_retrieved,
            total_relevant_count=total_relevant,
        )

    def batch_evaluate(
        self,
        retrieval_results: List[RetrievalResult],
        relevant_snippets_list: List[List[str]],
    ) -> List[EvaluationResult]:
        """Evaluate multiple retrieval results and return a list of metrics.

        Args:
            retrieval_results: Ordered list of retrieval outcomes.
            relevant_snippets_list: Parallel list of ground-truth snippet
                lists, one per entry in *retrieval_results*.

        Returns:
            A list of :class:`EvaluationResult` objects.
        """
        if len(retrieval_results) != len(relevant_snippets_list):
            raise ValueError(
                "retrieval_results and relevant_snippets_list must have the same length"
            )
        return [
            self.evaluate(result, snippets)
            for result, snippets in zip(retrieval_results, relevant_snippets_list)
        ]

    def aggregate_metrics(
        self, evaluation_results: List[EvaluationResult]
    ) -> dict:
        """Compute macro-averaged precision, recall and F1.

        Args:
            evaluation_results: A list of per-query evaluation results.

        Returns:
            A dict with keys ``"macro_precision"``, ``"macro_recall"`` and
            ``"macro_f1"``.
        """
        if not evaluation_results:
            return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

        n = len(evaluation_results)
        return {
            "macro_precision": sum(r.precision for r in evaluation_results) / n,
            "macro_recall": sum(r.recall for r in evaluation_results) / n,
            "macro_f1": sum(r.f1 for r in evaluation_results) / n,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_relevant(text: str, normalized_snippets: List[str]) -> bool:
        """Return *True* if *text* contains at least one relevant snippet."""
        text_lower = text.lower()
        return any(snippet in text_lower for snippet in normalized_snippets)
