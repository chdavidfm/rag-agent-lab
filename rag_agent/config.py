"""Runtime configuration, resolved from the environment.

Centralising this lookup keeps every module from inventing its own variable
names and defaults, and allows a local `.env` file to be loaded without
polluting the system environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load a .env file from the working directory when present. Variables already
# defined in the environment win: in production the real environment is the
# source of truth, not a file left behind on disk.
load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    """Execution parameters resolved from the environment."""

    docs_dir: Path
    backend: str
    embedding_model: str
    reranker_model: str
    rerank: bool
    cache_dir: Path
    chunk_size: int
    overlap: int
    top_k: int
    openai_api_key: str | None
    openai_base_url: str | None
    llm_model: str

    @property
    def llm_enabled(self) -> bool:
        """Whether credentials are available to compose answers with an LLM."""
        return bool(self.openai_api_key)


def _flag(name: str, default: str = "false") -> bool:
    """Read a boolean environment variable, tolerating the usual spellings."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    """Resolve settings from the environment at call time.

    Reading on each call, rather than once at import, lets tests adjust the
    environment without reloading the package.
    """
    return Settings(
        docs_dir=Path(os.getenv("RAG_DOCS_DIR", "data/sample")),
        backend=os.getenv("RAG_BACKEND", "tfidf").strip().lower(),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        reranker_model=os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        rerank=_flag("RAG_RERANK"),
        cache_dir=Path(os.getenv("RAG_CACHE_DIR", ".rag_cache")),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "500")),
        overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
        top_k=int(os.getenv("RAG_TOP_K", "3")),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        llm_model=os.getenv("RAG_MODEL", "gpt-4o-mini"),
    )
