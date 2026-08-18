"""Tests for settings resolved from the environment."""

from pathlib import Path

import pytest

from rag_agent.config import get_settings

VARIABLES = ("RAG_DOCS_DIR", "RAG_BACKEND", "RAG_TOP_K", "RAG_RERANK", "OPENAI_API_KEY")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for variable in VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_defaults():
    settings = get_settings()
    assert settings.docs_dir == Path("data/sample")
    assert settings.backend == "tfidf"
    assert settings.top_k == 3
    assert settings.rerank is False
    assert settings.llm_enabled is False


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("RAG_DOCS_DIR", "/tmp/docs")
    monkeypatch.setenv("RAG_BACKEND", "EMBEDDINGS")
    monkeypatch.setenv("RAG_TOP_K", "7")
    settings = get_settings()
    assert settings.docs_dir == Path("/tmp/docs")
    assert settings.backend == "embeddings"
    assert settings.top_k == 7


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_truthy_spellings_enable_reranking(monkeypatch, value):
    monkeypatch.setenv("RAG_RERANK", value)
    assert get_settings().rerank is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_other_spellings_leave_reranking_off(monkeypatch, value):
    monkeypatch.setenv("RAG_RERANK", value)
    assert get_settings().rerank is False


def test_a_key_enables_llm_mode(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_settings().llm_enabled is True


def test_an_empty_key_does_not(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert get_settings().llm_enabled is False
