"""Tests for the :mod:`financial_rag` module."""

import os
import sys

# Add the ``src`` directory to the Python path so tests can import the package
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from finqa.financial_rag import FinancialRAG


class DummyCrossEncoder:
    """Minimal cross-encoder used for testing re-ranking logic."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def predict(self, pairs):  # type: ignore[override]
        return [self.scores[doc] for _, doc in pairs]


def test_retrieve_returns_relevant_document() -> None:
    docs = [
        "Apple reported record revenue of $90B in Q1 2024",
        "Tesla's earnings grew 20% year over year",
    ]

    rag = FinancialRAG(docs)
    retrieved = rag.retrieve("Apple revenue", top_k=1)

    assert retrieved[0][0] == docs[0]


def test_answer_compiles_context() -> None:
    docs = ["Bank of America posted net income of $7B"]
    rag = FinancialRAG(docs)
    answer = rag.answer("How much did Bank of America earn?")

    assert "$7B" in answer


def test_cross_encoder_reranks_documents() -> None:
    docs = [
        "query query query noise",
        "query relevant information",
    ]
    ce = DummyCrossEncoder({docs[0]: 0.1, docs[1]: 0.9})
    rag = FinancialRAG(docs, cross_encoder=ce)

    retrieved = rag.retrieve("query", top_k=2)
    assert retrieved[0][0] == docs[1]


def test_hierarchical_retrieval_returns_sentence() -> None:
    docs = [
        "The first sentence. Revenue was $100B in 2023. Another note.",
        "Completely unrelated text.",
    ]
    rag = FinancialRAG(docs)

    sentences = rag.retrieve_hierarchical("Revenue", doc_top_k=1, sent_top_k=1)
    assert "Revenue was $100B" in sentences[0][0]


def test_hierarchical_answer_uses_sentences() -> None:
    docs = ["Net income was $7B. Cash flow increased." ]
    rag = FinancialRAG(docs)

    answer = rag.answer_hierarchical("net income", doc_top_k=1, sent_top_k=1)
    assert "$7B" in answer

