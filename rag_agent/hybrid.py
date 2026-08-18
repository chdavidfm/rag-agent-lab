"""Hybrid search through Reciprocal Rank Fusion.

The two retrieval strategies fail in different ways. Lexical search nails exact
terms — acronyms, identifiers, proper nouns — and misses synonyms. Dense search
captures meaning but can overlook a rare literal match. Combining them covers
each one's blind spot.

Their scores are not comparable: a cosine of 0.7 means nothing like a TF-IDF of
0.7. RRF sidesteps calibration entirely by discarding the scores and fusing only
the positions:

    score(d) = Σ  1 / (k + rank of d in list i)

A passage ranked highly by both strategies rises to the top; one found by a
single strategy still makes the cut, a little lower. This is the technique used
by modern search engines precisely because it needs no tuning.

Reference: Cormack, Clarke and Buettcher, SIGIR 2009.
"""

from __future__ import annotations

from typing import Any

from .retriever import Hit, Retriever

# Damping constant from the original paper. It softens the advantage of the top
# positions so a single first place cannot dominate the fusion.
RRF_K = 60


class HybridRetriever:
    """Fuses the rankings of several retrievers with RRF."""

    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], rrf_k: int = RRF_K) -> None:
        """Create the hybrid retriever.

        Args:
            retrievers: Already constructed retrievers to combine.
            rrf_k: Damping constant; larger values flatten the advantage of
                the leading positions.

        Raises:
            ValueError: If no retriever is supplied.
        """
        if not retrievers:
            raise ValueError("Hybrid search needs at least one retriever")
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def index(self, docs: list[str]) -> HybridRetriever:
        """Index the same corpus in every underlying retriever."""
        for retriever in self.retrievers:
            retriever.index(docs)
        return self

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Return the ``k`` best passages after fusing every ranking.

        Each retriever contributes more candidates than will be returned: a
        passage may sit fifth in one list and first in the other, and that
        signal would be lost by truncating too early.
        """
        depth = max(k * 3, 10)
        fused: dict[str, float] = {}

        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.search(query, k=depth), start=1):
                fused[hit.text] = fused.get(hit.text, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
        return [Hit(text=text, score=score) for text, score in ranked[:k]]

    def state_dict(self) -> dict[str, Any]:
        """Collect the state of every underlying retriever."""
        return {"members": [retriever.state_dict() for retriever in self.retrievers]}

    def load_state_dict(self, state: dict[str, Any]) -> HybridRetriever:
        """Restore each underlying retriever from its own saved state."""
        for retriever, member in zip(self.retrievers, state["members"], strict=True):
            retriever.load_state_dict(member)
        return self
