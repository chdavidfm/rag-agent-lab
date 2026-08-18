"""Tests for cross-encoder reranking.

A fake cross-encoder with a known scoring rule is injected, so what is verified
is the two-stage mechanics: how many candidates are requested, that the order
actually changes, and that the model is never loaded when it cannot help.
"""

import pytest

from rag_agent.rerank import RerankingRetriever
from rag_agent.retriever import Hit


class StubRetriever:
    """First stage returning a fixed list; records the depth it was asked for."""

    def __init__(self, texts):
        self.texts = texts
        self.requested_k = None
        self.indexed = None

    def index(self, docs):
        self.indexed = docs
        return self

    def search(self, query, k=3):
        self.requested_k = k
        return [Hit(text=t, score=1.0 - i * 0.1) for i, t in enumerate(self.texts[:k])]

    def state_dict(self):
        return {"texts": self.texts}

    def load_state_dict(self, state):
        self.texts = list(state["texts"])
        return self


class KeywordCrossEncoder:
    """Scores a pair by how often the query word occurs in the passage."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return [passage.lower().count(query.lower()) for query, passage in pairs]


def test_reranking_promotes_the_better_passage():
    """The first stage ranks 'target' last; the cross-encoder lifts it."""
    base = StubRetriever(["noise one", "noise two", "target target target"])
    reranked = RerankingRetriever(base, model=KeywordCrossEncoder()).search("target", k=1)
    assert reranked[0].text == "target target target"


def test_requests_more_candidates_than_it_returns():
    """Reranking is worthless if the first stage never surfaces the answer."""
    base = StubRetriever([f"passage {i}" for i in range(30)])
    RerankingRetriever(base, model=KeywordCrossEncoder(), depth=5).search("passage", k=3)
    assert base.requested_k == 15


def test_returns_exactly_k_results():
    base = StubRetriever([f"passage {i}" for i in range(20)])
    hits = RerankingRetriever(base, model=KeywordCrossEncoder()).search("passage", k=4)
    assert len(hits) == 4


def test_scores_come_from_the_cross_encoder():
    base = StubRetriever(["aa", "a"])
    hits = RerankingRetriever(base, model=KeywordCrossEncoder()).search("a", k=2)
    assert [hit.score for hit in hits] == [2.0, 1.0]


def test_results_are_ordered_by_the_new_scores():
    base = StubRetriever(["a", "aaa", "aa"])
    hits = RerankingRetriever(base, model=KeywordCrossEncoder()).search("a", k=3)
    assert [hit.score for hit in hits] == sorted([hit.score for hit in hits], reverse=True)


def test_a_single_candidate_skips_the_model():
    """With nothing to reorder, the expensive model must not be loaded."""
    encoder = KeywordCrossEncoder()
    hits = RerankingRetriever(StubRetriever(["only"]), model=encoder).search("q", k=1)
    assert hits[0].text == "only"
    assert encoder.calls == 0


def test_depth_must_be_positive():
    with pytest.raises(ValueError, match="depth"):
        RerankingRetriever(StubRetriever([]), model=KeywordCrossEncoder(), depth=0)


def test_indexing_reaches_the_first_stage():
    base = StubRetriever(["a"])
    RerankingRetriever(base, model=KeywordCrossEncoder()).index(["doc"])
    assert base.indexed == ["doc"]


def test_state_is_delegated_to_the_first_stage():
    base = StubRetriever(["a"])
    wrapper = RerankingRetriever(base, model=KeywordCrossEncoder())
    restored = RerankingRetriever(StubRetriever([]), model=KeywordCrossEncoder())
    restored.load_state_dict(wrapper.state_dict())
    assert restored.base.texts == ["a"]


def test_satisfies_the_retriever_contract():
    from rag_agent.retriever import Retriever

    assert isinstance(RerankingRetriever(StubRetriever([]), model=KeywordCrossEncoder()), Retriever)
