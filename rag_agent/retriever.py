"""Retrieval of the passages most relevant to a query.

Defines the contract every retriever satisfies (`Retriever`) and the lexical
implementation built on TF-IDF. Programming against the contract, rather than a
concrete class, lets the search strategy change without touching the pipeline,
the API or the evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Hit:
    """A retrieved passage together with its relevance score."""

    text: str
    score: float


@runtime_checkable
class Retriever(Protocol):
    """The minimal contract of a retriever.

    Any object that can index a list of passages and return those closest to a
    query fits here, with no inheritance required.
    """

    def index(self, docs: list[str]) -> Retriever:
        """Build the search structure over ``docs``."""
        ...

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Return the ``k`` most relevant passages, best first."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Return the state needed to restore the index without rebuilding it."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> Retriever:
        """Restore a previously saved state."""
        ...


class TfidfRetriever:
    """Lexical retrieval: TF-IDF vectors compared by cosine similarity.

    Fast, with no model to download, and dependable whenever the question
    shares vocabulary with the documents. Its limit is that it cannot connect
    synonyms; that is what the dense retriever is for.
    """

    name = "tfidf"

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None
        self._docs: list[str] = []

    def index(self, docs: list[str]) -> TfidfRetriever:
        """Fit the vectoriser over ``docs`` and store the resulting matrix."""
        if not docs:
            raise ValueError("Cannot index an empty document list")
        self._docs = docs
        self._matrix = self._vectorizer.fit_transform(docs)
        return self

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Return the ``k`` passages sharing the most distinctive terms."""
        if self._matrix is None:
            raise RuntimeError("index() must be called before search()")
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        ranked = sorted(
            zip(self._docs, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [Hit(text=text, score=float(score)) for text, score in ranked[:k]]

    def state_dict(self) -> dict[str, Any]:
        """Return the passages; refitting TF-IDF from them is inexpensive."""
        return {"docs": self._docs}

    def load_state_dict(self, state: dict[str, Any]) -> TfidfRetriever:
        """Rebuild the index from stored passages."""
        return self.index(list(state["docs"]))
