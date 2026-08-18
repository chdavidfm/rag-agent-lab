"""Selección del recuperador a usar.

Un único punto donde se traduce el nombre de un backend a su
implementación. Añadir una estrategia nueva es registrarla aquí; el resto
del sistema no se entera.
"""

from __future__ import annotations

from .retriever import Retriever, TfidfRetriever

BACKENDS = ("tfidf", "embeddings")


def build_retriever(backend: str = "tfidf", *, model_name: str | None = None) -> Retriever:
    """Construye el recuperador indicado por `backend`.

    Args:
        backend: "tfidf" (léxico) o "embeddings" (denso).
        model_name: Modelo de embeddings, solo para el backend denso.

    Raises:
        ValueError: Si el backend no existe.
    """
    normalized = backend.strip().lower()
    if normalized == "tfidf":
        return TfidfRetriever()
    if normalized == "embeddings":
        # Import local: mantiene la dependencia pesada fuera del arranque.
        from .embeddings import DEFAULT_MODEL, EmbeddingRetriever

        return EmbeddingRetriever(model_name or DEFAULT_MODEL)
    raise ValueError(f"Backend desconocido: {backend!r}. Opciones: {', '.join(BACKENDS)}")
