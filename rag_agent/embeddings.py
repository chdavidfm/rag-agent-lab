"""Recuperación densa mediante embeddings neuronales.

A diferencia de TF-IDF, que compara palabras, este recuperador compara
*significados*: convierte cada fragmento en un vector y busca los más
cercanos a la pregunta. Así encuentra respuestas aunque no compartan
vocabulario ("coche" frente a "automóvil").

El modelo se carga de forma perezosa: `sentence-transformers` arrastra
PyTorch (cientos de MB), y quien solo use el modo léxico no debería pagar
esa descarga.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .retriever import Hit

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones de tipo
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingRetriever:
    """Recuperador denso basado en similitud coseno entre embeddings."""

    name = "embeddings"

    def __init__(self, model_name: str = DEFAULT_MODEL, model: Any | None = None) -> None:
        """Crea el recuperador.

        Args:
            model_name: Identificador del modelo en Hugging Face.
            model: Modelo ya cargado. Permite inyectar un doble en tests o
                reutilizar una instancia entre recuperadores, que es lo caro.
        """
        self.model_name = model_name
        self._model = model
        self._docs: list[str] = []
        self._embeddings: np.ndarray | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Carga el modelo la primera vez que hace falta."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depende del entorno
                raise ImportError(
                    "El recuperador denso necesita 'sentence-transformers'. "
                    'Instálalo con: pip install "rag-agent-lab[embeddings]"'
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Codifica textos como vectores unitarios.

        Al normalizar a norma 1, la similitud coseno se reduce a un producto
        escalar: una multiplicación de matrices en lugar de recorrer pares.
        """
        vectors = np.asarray(self.model.encode(texts), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Evita dividir por cero si algún vector fuese nulo.
        return vectors / np.maximum(norms, 1e-12)

    def index(self, docs: list[str]) -> EmbeddingRetriever:
        """Calcula y guarda el embedding de cada fragmento."""
        if not docs:
            raise ValueError("No hay documentos que indexar")
        self._docs = docs
        self._embeddings = self._encode(docs)
        return self

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Devuelve los `k` fragmentos semánticamente más cercanos."""
        if self._embeddings is None:
            raise RuntimeError("Debes llamar a index() antes de search()")
        scores = self._encode([query]) @ self._embeddings.T
        scores = scores[0]
        # argsort ordena de menor a mayor: se toman los k últimos y se invierten.
        top = np.argsort(scores)[::-1][:k]
        return [Hit(text=self._docs[i], score=float(scores[i])) for i in top]
