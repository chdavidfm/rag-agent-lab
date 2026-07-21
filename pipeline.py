"""Orquestar el flujo RAG completo.

ingesta -> trocear -> indexar -> recuperar -> generar

Esta clase junta las piezas de los otros módulos en un único objeto fácil
de usar tanto desde la CLI como desde otro programa.
"""

from __future__ import annotations

from pathlib import Path

from .chunk import chunk_text
from .generate import answer
from .retriever import TfidfRetriever


class RagPipeline:
    """Pipeline RAG mínimo, de principio a fin."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50, k: int = 3) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.k = k
        self.retriever = TfidfRetriever()

    def index_paths(self, paths: list[Path]) -> "RagPipeline":
        """Lee y trocea cada archivo, y construye el índice de recuperación."""
        chunks: list[str] = []
        for path in paths:
            text = Path(path).read_text(encoding="utf-8")
            chunks.extend(chunk_text(text, self.chunk_size, self.overlap))
        self.retriever.index(chunks)
        return self

    def ask(self, question: str) -> str:
        """Recupera el contexto relevante y genera la respuesta."""
        hits = self.retriever.search(question, k=self.k)
        return answer(question, [hit.text for hit in hits])
