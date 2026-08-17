"""Trocear documentos en fragmentos ('chunks') solapados.

Es el primer paso de cualquier RAG: un documento largo se parte en trozos
manejables. El solape evita perder contexto justo en los bordes de un trozo.
Este módulo es Python puro (sin dependencias) para que sea fácil de leer.
"""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Divide ``text`` en fragmentos de ~``chunk_size`` caracteres con solape.

    Args:
        text: El texto de entrada.
        chunk_size: Tamaño máximo (en caracteres) de cada fragmento.
        overlap: Cuántos caracteres se repiten entre fragmentos consecutivos.

    Returns:
        Lista de fragmentos no vacíos.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap debe estar en el rango [0, chunk_size)")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        fragment = text[start : start + chunk_size].strip()
        if fragment:
            chunks.append(fragment)
        start += step
    return chunks
