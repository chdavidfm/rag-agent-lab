"""Tests de la configuración leída del entorno."""

from pathlib import Path

from rag_agent.config import get_settings


def test_valores_por_defecto(monkeypatch):
    for var in ("RAG_DOCS_DIR", "RAG_BACKEND", "RAG_TOP_K", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    settings = get_settings()
    assert settings.docs_dir == Path("data/sample")
    assert settings.backend == "tfidf"
    assert settings.top_k == 3
    assert settings.llm_enabled is False


def test_el_entorno_sobrescribe_los_valores(monkeypatch):
    monkeypatch.setenv("RAG_DOCS_DIR", "/tmp/docs")
    monkeypatch.setenv("RAG_BACKEND", "EMBEDDINGS")
    monkeypatch.setenv("RAG_TOP_K", "7")
    settings = get_settings()
    assert settings.docs_dir == Path("/tmp/docs")
    assert settings.backend == "embeddings"
    assert settings.top_k == 7


def test_la_clave_activa_el_modo_llm(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
    assert get_settings().llm_enabled is True


def test_una_clave_vacia_no_activa_el_modo_llm(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert get_settings().llm_enabled is False
