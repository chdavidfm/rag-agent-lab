"""Orquestación del flujo RAG completo.

ingesta -> trocear -> indexar -> recuperar -> generar

Reúne las etapas en un único objeto, igual de cómodo desde la CLI, desde la
API o desde el banco de evaluación. El recuperador es intercambiable: el
pipeline solo conoce el contrato `Retriever`.
"""

from __future__ import annotations

from pathlib import Path

from .chunk import chunk_text
from .factory import build_retriever
from .generate import answer
from .retriever import Hit, Retriever


class RagPipeline:
    """Pipeline RAG de principio a fin."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        k: int = 3,
        retriever: Retriever | None = None,
        backend: str = "tfidf",
    ) -> None:
        """Crea el pipeline.

        Args:
            chunk_size: Tamaño de cada fragmento, en caracteres.
            overlap: Solape entre fragmentos consecutivos.
            k: Fragmentos a recuperar por consulta.
            retriever: Recuperador ya construido; tiene prioridad sobre
                `backend` y permite inyectar uno propio o un doble de test.
            backend: Estrategia a construir si no se pasa `retriever`.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.k = k
        self.retriever = retriever if retriever is not None else build_retriever(backend)

    def index_paths(self, paths: list[Path]) -> RagPipeline:
        """Lee y trocea cada archivo, y construye el índice de recuperación."""
        chunks: list[str] = []
        for path in paths:
            try:
                text = Path(path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ValueError(f"No se pudo leer el documento {path}: {exc}") from exc
            chunks.extend(chunk_text(text, self.chunk_size, self.overlap))
        if not chunks:
            raise ValueError("Los documentos no contienen texto que indexar")
        self.retriever.index(chunks)
        return self

    def ask(self, question: str) -> str:
        """Recupera el contexto relevante y genera la respuesta."""
        return self.answer_from(question, self.retriever.search(question, k=self.k))

    def answer_from(self, question: str, hits: list[Hit]) -> str:
        """Genera la respuesta a partir de unos fragmentos ya recuperados.

        Permite a quien ya tiene los fragmentos (por ejemplo la API, que los
        devuelve junto a la respuesta) evitar una segunda búsqueda.
        """
        return answer(question, [hit.text for hit in hits])
