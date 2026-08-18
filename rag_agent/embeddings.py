"""Dense retrieval through neural sentence embeddings.

Where TF-IDF compares words, this retriever compares meaning: each passage
becomes a vector, and the closest vectors to the question win. That surfaces an
answer about a feline when the question mentions a cat, even though the two
words never coincide.

The model is loaded lazily. `sentence-transformers` pulls in PyTorch, hundreds
of megabytes, and nobody using the lexical backend should pay that cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .retriever import Hit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Import and instantiate a sentence-transformers model.

    Raises:
        ImportError: With installation guidance when the extra is missing.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Dense retrieval requires 'sentence-transformers'. "
            'Install it with: pip install "rag-agent-lab[embeddings]"'
        ) from exc
    return SentenceTransformer(model_name)


class EmbeddingRetriever:
    """Dense retrieval scored by cosine similarity between embeddings."""

    name = "embeddings"

    def __init__(self, model_name: str = DEFAULT_MODEL, model: Any | None = None) -> None:
        """Create the retriever.

        Args:
            model_name: Hugging Face identifier of the encoder.
            model: An already loaded encoder. Useful to inject a double in
                tests, or to share one instance across retrievers, since
                loading is the expensive part.
        """
        self.model_name = model_name
        self._model = model
        self._docs: list[str] = []
        self._embeddings: np.ndarray | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Load the encoder the first time it is actually needed."""
        if self._model is None:
            self._model = load_sentence_transformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts as unit vectors.

        Normalising to length one turns cosine similarity into a plain dot
        product, so a whole corpus is scored with a single matrix multiply.
        """
        vectors = np.asarray(self.model.encode(texts), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Guard against a zero vector, which would otherwise divide by zero.
        return vectors / np.maximum(norms, 1e-12)

    def index(self, docs: list[str]) -> EmbeddingRetriever:
        """Encode and store one embedding per passage."""
        if not docs:
            raise ValueError("Cannot index an empty document list")
        self._docs = docs
        self._embeddings = self._encode(docs)
        return self

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Return the ``k`` passages closest in meaning to ``query``."""
        if self._embeddings is None:
            raise RuntimeError("index() must be called before search()")
        scores = (self._encode([query]) @ self._embeddings.T)[0]
        # argsort is ascending, so take the tail and reverse it.
        top = np.argsort(scores)[::-1][:k]
        return [Hit(text=self._docs[i], score=float(scores[i])) for i in top]

    def state_dict(self) -> dict[str, Any]:
        """Return passages and vectors, so nothing has to be re-encoded."""
        if self._embeddings is None:
            raise RuntimeError("index() must be called before state_dict()")
        return {"docs": self._docs, "embeddings": self._embeddings, "model": self.model_name}

    def load_state_dict(self, state: dict[str, Any]) -> EmbeddingRetriever:
        """Restore passages and vectors without touching the encoder."""
        self._docs = list(state["docs"])
        self._embeddings = np.asarray(state["embeddings"], dtype=np.float32)
        return self
