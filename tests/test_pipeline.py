"""Tests for the end-to-end pipeline, including index persistence."""

import pytest

from rag_agent.pipeline import RagPipeline
from rag_agent.retriever import TfidfRetriever


@pytest.fixture(autouse=True)
def local_mode(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "retrieval.txt").write_text(
        "RAG combines retrieval of documents with text generation.", encoding="utf-8"
    )
    (tmp_path / "cooking.txt").write_text(
        "Paella is cooked with rice, saffron and olive oil.", encoding="utf-8"
    )
    return sorted(tmp_path.glob("*.txt"))


def test_retrieves_from_the_document_about_the_topic(corpus):
    pipeline = RagPipeline(k=1).index_paths(corpus)
    assert "Paella" in pipeline.retriever.search("What goes into paella?", k=1)[0].text


def test_ask_returns_a_grounded_answer(corpus):
    answer = RagPipeline(k=1).index_paths(corpus).ask("What is RAG?")
    assert "retrieval" in answer.lower()


def test_answer_from_reuses_the_given_passages(corpus):
    pipeline = RagPipeline().index_paths(corpus)
    hits = pipeline.retriever.search("paella", k=1)
    assert "Paella" in pipeline.answer_from("paella", hits)


def test_indexing_nothing_fails():
    with pytest.raises(ValueError):
        RagPipeline().index_paths([])


def test_an_unreadable_document_is_reported_with_its_path(tmp_path):
    broken = tmp_path / "broken.txt"
    broken.write_bytes(b"\xff\xfe invalid utf-8 \xff")
    with pytest.raises(ValueError, match="broken.txt"):
        RagPipeline().index_paths([broken])


def test_a_custom_retriever_takes_precedence(corpus):
    injected = TfidfRetriever()
    assert RagPipeline(retriever=injected).index_paths(corpus).retriever is injected


# --- Index cache ----------------------------------------------------------


def test_the_first_run_builds_and_the_second_reuses(corpus, tmp_path):
    cache = tmp_path / "cache"

    first = RagPipeline(cache_dir=cache).index_paths(corpus)
    assert first.loaded_from_cache is False

    second = RagPipeline(cache_dir=cache).index_paths(corpus)
    assert second.loaded_from_cache is True


def test_the_reused_index_returns_the_same_results(corpus, tmp_path):
    cache = tmp_path / "cache"
    question = "What is RAG?"

    built = RagPipeline(cache_dir=cache).index_paths(corpus).ask(question)
    reused = RagPipeline(cache_dir=cache).index_paths(corpus).ask(question)
    assert built == reused


def test_editing_a_document_invalidates_the_cache(corpus, tmp_path):
    cache = tmp_path / "cache"
    RagPipeline(cache_dir=cache).index_paths(corpus)

    corpus[0].write_text("Completely different content now.", encoding="utf-8")
    assert RagPipeline(cache_dir=cache).index_paths(corpus).loaded_from_cache is False


def test_changing_the_chunk_size_invalidates_the_cache(corpus, tmp_path):
    cache = tmp_path / "cache"
    RagPipeline(cache_dir=cache, chunk_size=500).index_paths(corpus)
    assert (
        RagPipeline(cache_dir=cache, chunk_size=120).index_paths(corpus).loaded_from_cache is False
    )


def test_without_a_cache_directory_nothing_is_persisted(corpus, tmp_path):
    pipeline = RagPipeline(cache_dir=None).index_paths(corpus)
    assert pipeline.loaded_from_cache is False
    assert not list(tmp_path.glob("*.json"))
