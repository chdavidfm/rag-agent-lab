"""Assembling the retrieval stack from configuration.

A single place translates backend names into implementations and wraps them
with the optional reranking stage. Registering a new strategy here is all it
takes; the rest of the system keeps talking to the `Retriever` contract.
"""

from __future__ import annotations

from .retriever import Retriever, TfidfRetriever

BACKENDS = ("tfidf", "embeddings", "hybrid")


def build_retriever(
    backend: str = "tfidf",
    *,
    model_name: str | None = None,
    rerank: bool = False,
    reranker_model: str | None = None,
) -> Retriever:
    """Build the retrieval stack described by the arguments.

    Args:
        backend: "tfidf" (lexical), "embeddings" (dense) or "hybrid" (both,
            fused with Reciprocal Rank Fusion).
        model_name: Encoder for the dense backend; ignored by the lexical one.
        rerank: Whether to add a cross-encoder second stage on top.
        reranker_model: Cross-encoder to use when reranking.

    Returns:
        A retriever satisfying the `Retriever` contract.

    Raises:
        ValueError: If the backend is unknown.
    """
    base = _build_base(backend, model_name)
    if not rerank:
        return base

    # Imported here so the reranker never loads for callers that do not use it.
    from .rerank import DEFAULT_RERANKER, RerankingRetriever

    return RerankingRetriever(base, reranker_model or DEFAULT_RERANKER)


def _build_base(backend: str, model_name: str | None) -> Retriever:
    """Construct the first-stage retriever."""
    normalized = backend.strip().lower()

    if normalized == "tfidf":
        return TfidfRetriever()

    if normalized == "embeddings":
        return _build_dense(model_name)

    if normalized == "hybrid":
        from .hybrid import HybridRetriever

        return HybridRetriever([TfidfRetriever(), _build_dense(model_name)])

    raise ValueError(f"Unknown backend: {backend!r}. Available: {', '.join(BACKENDS)}")


def _build_dense(model_name: str | None) -> Retriever:
    """Construct the dense retriever, importing it only when required."""
    from .embeddings import DEFAULT_MODEL, EmbeddingRetriever

    return EmbeddingRetriever(model_name or DEFAULT_MODEL)
