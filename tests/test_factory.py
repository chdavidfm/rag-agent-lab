"""Tests for assembling the retrieval stack."""

import pytest

from rag_agent.factory import build_retriever
from rag_agent.retriever import Retriever, TfidfRetriever


def test_builds_the_lexical_retriever():
    assert isinstance(build_retriever("tfidf"), TfidfRetriever)


def test_accepts_padded_and_uppercase_names():
    assert isinstance(build_retriever("  TFIDF "), TfidfRetriever)


def test_an_unknown_backend_reports_the_alternatives():
    with pytest.raises(ValueError, match="Unknown backend"):
        build_retriever("magic")


def test_reranking_wraps_the_chosen_backend():
    from rag_agent.rerank import RerankingRetriever

    stack = build_retriever("tfidf", rerank=True)
    assert isinstance(stack, RerankingRetriever)
    assert isinstance(stack.base, TfidfRetriever)


def test_without_reranking_the_backend_is_returned_bare():
    assert isinstance(build_retriever("tfidf", rerank=False), TfidfRetriever)


def test_every_stack_satisfies_the_contract():
    assert isinstance(build_retriever("tfidf"), Retriever)
    assert isinstance(build_retriever("tfidf", rerank=True), Retriever)
