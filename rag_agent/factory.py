"""Selección de la estrategia de recuperación.

Un único punto donde el nombre de un backend se traduce a su
implementación. Añadir una estrategia nueva es registrarla aquí; el resto
del sistema sigue hablando con el contrato `Retriever`.
"""

from __future__ import annotations

from .retriever import Retriever, TfidfRetriever

BACKENDS = ("tfidf", "embeddings", "hybrid")


def build_retriever(backend: str = "tfidf", *, model_name: str | None = None) -> Retriever:
    """Construye el recuperador indicado por `backend`.

    Args:
        backend: "tfidf" (léxico), "embeddings" (semántico) o "hybrid"
            (fusión de ambos mediante Reciprocal Rank Fusion).
        model_name: Modelo de embeddings; se ignora en el backend léxico.

    Raises:
        ValueError: Si el backend no existe.
    """
    normalized = backend.strip().lower()

    if normalized == "tfidf":
        return TfidfRetriever()

    if normalized == "embeddings":
        return _build_embedding_retriever(model_name)

    if normalized == "hybrid":
        # Import local: la fusión solo se carga si se pide.
        from .hybrid import HybridRetriever

        return HybridRetriever([TfidfRetriever(), _build_embedding_retriever(model_name)])

    raise ValueError(f"Backend desconocido: {backend!r}. Opciones: {', '.join(BACKENDS)}")


def _build_embedding_retriever(model_name: str | None) -> Retriever:
    """Crea el recuperador denso, importándolo solo cuando hace falta."""
    from .embeddings import DEFAULT_MODEL, EmbeddingRetriever

    return EmbeddingRetriever(model_name or DEFAULT_MODEL)
