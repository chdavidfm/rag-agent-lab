"""Configuración del agente, leída del entorno.

Centralizar aquí la lectura de variables evita que cada módulo invente su
propio nombre o su propio valor por defecto, y permite cargar un archivo
`.env` local sin ensuciar el entorno del sistema.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Carga el .env del directorio de trabajo si existe. Las variables que ya
# estén definidas en el entorno tienen prioridad: en producción manda el
# entorno real, no un archivo olvidado en el disco.
load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    """Parámetros de ejecución resueltos desde el entorno."""

    docs_dir: Path
    backend: str
    embedding_model: str
    chunk_size: int
    overlap: int
    top_k: int
    openai_api_key: str | None
    openai_base_url: str | None
    llm_model: str

    @property
    def llm_enabled(self) -> bool:
        """Hay credenciales para redactar la respuesta con un LLM."""
        return bool(self.openai_api_key)


def get_settings() -> Settings:
    """Construye los ajustes leyendo el entorno en el momento de la llamada.

    Se resuelve en cada llamada (y no en una constante de módulo) para que
    los tests puedan modificar el entorno sin reimportar el paquete.
    """
    return Settings(
        docs_dir=Path(os.getenv("RAG_DOCS_DIR", "data/sample")),
        backend=os.getenv("RAG_BACKEND", "tfidf").strip().lower(),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "500")),
        overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
        top_k=int(os.getenv("RAG_TOP_K", "3")),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        llm_model=os.getenv("RAG_MODEL", "gpt-4o-mini"),
    )
