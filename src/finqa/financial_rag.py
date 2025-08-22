"""Basic financial retrieval-augmented generation (RAG) system.

This module provides a small example of how to build a RAG style
question answering system over financial documents.  It uses
``scikit-learn`` for TF–IDF based document retrieval and a very simple
template based generation step.  The goal is to offer an easily
understandable starting point that can be expanded with more advanced
models.

Example
-------
>>> rag = FinancialRAG([
...     "Apple reported record revenue of $90B in Q1 2024",
...     "Tesla's earnings grew 20% year over year"
... ])
>>> rag.answer("How much revenue did Apple report in Q1 2024?")
'Based on the documents: Apple reported record revenue of $90B in Q1 2024'
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class FinancialRAG:
    """Tiny retrieval augmented generation helper.

    Parameters
    ----------
    documents:
        A list of textual financial documents.  These could be earnings
        call transcripts, SEC filings, news articles, etc.
    """

    documents: List[str] = field(default_factory=list)
    cross_encoder: Optional[Any] = None

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer()
        if self.documents:
            self.doc_vectors = self.vectorizer.fit_transform(self.documents)
        else:
            self.doc_vectors = None

        # Allow passing a model name to lazily load a CrossEncoder if
        # ``sentence-transformers`` is available.  In tests we inject a
        # lightweight object with a ``predict`` method instead of loading a
        # heavy model.
        if isinstance(self.cross_encoder, str):
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self.cross_encoder = CrossEncoder(self.cross_encoder)
            except Exception:  # pragma: no cover - optional dependency
                self.cross_encoder = None

    # ------------------------------------------------------------------
    def add_documents(self, docs: List[str]) -> None:
        """Add new documents to the retrieval index."""

        self.documents.extend(docs)
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3, *, candidate_multiplier: int = 3) -> List[Tuple[str, float]]:
        """Return the ``top_k`` most similar documents.

        A simple TF–IDF similarity search is used to generate candidate
        documents.  If a cross-encoder is supplied, those candidates are
        re-ranked using the model's scores.

        Parameters
        ----------
        query:
            Natural language question describing the desired financial
            information.
        top_k:
            Number of documents to return.
        candidate_multiplier:
            When using a cross-encoder we first retrieve ``top_k`` ×
            ``candidate_multiplier`` documents using TF–IDF and then
            re-rank them.
        """

        if not self.documents:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_ids = scores.argsort()[::-1][: top_k * candidate_multiplier]
        candidates = [self.documents[i] for i in top_ids]

        if self.cross_encoder:
            pairs = [[query, doc] for doc in candidates]
            ce_scores = self.cross_encoder.predict(pairs)
            ranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
            return [(doc, float(score)) for doc, score in ranked[:top_k]]

        return [(self.documents[i], float(scores[i])) for i in top_ids[:top_k]]

    # ------------------------------------------------------------------
    def retrieve_hierarchical(
        self,
        query: str,
        doc_top_k: int = 3,
        sent_top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Two stage retrieval returning fine-grained text snippets.

        The method first finds the most relevant documents and then searches
        within those documents for the most relevant sentences.  A
        cross-encoder is optionally used to refine the final sentence ranking.
        """

        top_docs = self.retrieve(query, top_k=doc_top_k)
        if not top_docs:
            return []

        sentences: List[str] = []
        for doc, _ in top_docs:
            sentences.extend(
                s.strip()
                for s in re.split(r"(?<=[.!?]) +", doc)
                if s.strip()
            )

        if not sentences:
            return []

        vec = TfidfVectorizer()
        sent_vecs = vec.fit_transform(sentences)
        query_vec = vec.transform([query])
        scores = cosine_similarity(query_vec, sent_vecs).flatten()
        top_ids = scores.argsort()[::-1][:sent_top_k]
        snippets = [sentences[i] for i in top_ids]

        if self.cross_encoder:
            pairs = [[query, s] for s in snippets]
            ce_scores = self.cross_encoder.predict(pairs)
            ranked = sorted(zip(snippets, ce_scores), key=lambda x: x[1], reverse=True)
            return [(text, float(score)) for text, score in ranked]

        return [(snippets[i], float(scores[top_ids[i]])) for i in range(len(top_ids))]

    # ------------------------------------------------------------------
    def answer(self, query: str, top_k: int = 3) -> str:
        """Generate a simple answer using retrieved document context."""

        retrieved = self.retrieve(query, top_k)
        if not retrieved:
            return "No documents available."

        context = " ".join(doc for doc, _ in retrieved)
        return f"Based on the documents: {context}"

    # ------------------------------------------------------------------
    def answer_hierarchical(
        self, query: str, doc_top_k: int = 3, sent_top_k: int = 3
    ) -> str:
        """Generate an answer using sentence level retrieval."""

        retrieved = self.retrieve_hierarchical(query, doc_top_k, sent_top_k)
        if not retrieved:
            return "No documents available."

        context = " ".join(text for text, _ in retrieved)
        return f"Based on the documents: {context}"


__all__ = ["FinancialRAG"]

