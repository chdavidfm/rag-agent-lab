"""Tests for the lexical retriever."""

import pytest

from rag_agent.retriever import Retriever, TfidfRetriever

DOCS = [
    "Cats are domestic feline mammals.",
    "Python is a widely used programming language.",
    "RAG combines retrieval of information with text generation.",
]


@pytest.fixture
def retriever():
    return TfidfRetriever().index(DOCS)


def test_ranks_the_relevant_document_first(retriever):
    assert "RAG" in retriever.search("What is RAG?", k=1)[0].text


def test_returns_results_in_descending_score(retriever):
    scores = [hit.score for hit in retriever.search("programming language", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_never_returns_more_than_requested(retriever):
    assert len(retriever.search("cats", k=2)) == 2


def test_searching_before_indexing_fails():
    with pytest.raises(RuntimeError):
        TfidfRetriever().search("anything")


def test_indexing_nothing_fails():
    with pytest.raises(ValueError):
        TfidfRetriever().index([])


def test_satisfies_the_retriever_contract(retriever):
    assert isinstance(retriever, Retriever)


def test_state_round_trip_preserves_results(retriever):
    restored = TfidfRetriever().load_state_dict(retriever.state_dict())
    assert (
        restored.search("What is RAG?", k=1)[0].text
        == retriever.search("What is RAG?", k=1)[0].text
    )
