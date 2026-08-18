"""Second-stage reranking with a cross-encoder.

First-stage retrieval encodes the query and each passage independently, which
is what makes it fast: every passage is embedded once, ahead of time. The cost
is precision, because the model never sees the pair together and cannot judge
how the passage answers this particular question.

A cross-encoder reads query and passage jointly and scores their relevance
directly. It is far more accurate and far too slow to run over an entire
collection, so production systems use both: retrieve a generous candidate set
cheaply, then rerank a handful precisely.

    query ──► retrieve top 20 (fast) ──► rerank to top 3 (accurate) ──► answer

Reference: Nogueira and Cho, "Passage Re-ranking with BERT" (2019).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .retriever import Hit, Retriever

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import CrossEncoder

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates the first stage supplies per result finally returned.
# Too few starves the reranker; too many wastes its time, since it is the
# expensive part of the pipeline.
DEFAULT_DEPTH = 5


def load_cross_encoder(model_name: str) -> CrossEncoder:
    """Import and instantiate a cross-encoder model.

    Raises:
        ImportError: With installation guidance when the extra is missing.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reranking requires 'sentence-transformers'. "
            'Install it with: pip install "rag-agent-lab[embeddings]"'
        ) from exc
    return CrossEncoder(model_name)


class RerankingRetriever:
    """Wraps a retriever and reorders its candidates with a cross-encoder.

    It satisfies the same `Retriever` contract as the stage it wraps, so the
    pipeline, the API and the evaluation harness treat it identically.
    """

    name = "rerank"

    def __init__(
        self,
        base: Retriever,
        model_name: str = DEFAULT_RERANKER,
        model: Any | None = None,
        depth: int = DEFAULT_DEPTH,
    ) -> None:
        """Create the reranking stage.

        Args:
            base: First-stage retriever supplying the candidates.
            model_name: Hugging Face identifier of the cross-encoder.
            model: An already loaded cross-encoder, for tests or reuse.
            depth: Candidates requested per result returned.

        Raises:
            ValueError: If ``depth`` is not at least 1.
        """
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.base = base
        self.model_name = model_name
        self._model = model
        self.depth = depth

    @property
    def model(self) -> CrossEncoder:
        """Load the cross-encoder the first time it is actually needed."""
        if self._model is None:
            self._model = load_cross_encoder(self.model_name)
        return self._model

    def index(self, docs: list[str]) -> RerankingRetriever:
        """Index the corpus in the first-stage retriever."""
        self.base.index(docs)
        return self

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Retrieve a candidate set, then return the ``k`` best after scoring.

        When the first stage returns a single candidate there is nothing to
        reorder, so the cross-encoder is skipped and the model is never loaded.
        """
        candidates = self.base.search(query, k=k * self.depth)
        if len(candidates) <= 1:
            return candidates[:k]

        scores = self.model.predict([(query, hit.text) for hit in candidates])
        reranked = sorted(
            (
                Hit(text=hit.text, score=float(score))
                for hit, score in zip(candidates, scores, strict=True)
            ),
            key=lambda hit: hit.score,
            reverse=True,
        )
        return reranked[:k]

    def state_dict(self) -> dict[str, Any]:
        """Delegate persistence to the first-stage retriever."""
        return {"base": self.base.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> RerankingRetriever:
        """Restore the first-stage retriever; the reranker holds no index."""
        self.base.load_state_dict(state["base"])
        return self
