"""Tests for Reciprocal Rank Fusion.

Retrievers with known, fixed rankings are combined, so what is verified is the
fusion formula itself rather than the behaviour of any model.
"""

import pytest

from rag_agent.hybrid import RRF_K, HybridRetriever
from rag_agent.retriever import Hit


class FixedRanking:
    """A retriever that always returns the same ordering."""

    def __init__(self, texts):
        self.texts = texts
        self.indexed = None

    def index(self, docs):
        self.indexed = docs
        return self

    def search(self, query, k=3):
        return [Hit(text=t, score=1.0 - i * 0.1) for i, t in enumerate(self.texts[:k])]

    def state_dict(self):
        return {"texts": self.texts}

    def load_state_dict(self, state):
        self.texts = list(state["texts"])
        return self


def test_agreement_between_strategies_wins():
    lexical = FixedRanking(["agreed", "lexical-only", "filler"])
    dense = FixedRanking(["agreed", "dense-only", "filler"])
    assert HybridRetriever([lexical, dense]).search("q", k=3)[0].text == "agreed"


def test_rescues_what_only_one_strategy_finds():
    """The whole point of fusion: covering the other strategy's blind spot."""
    lexical = FixedRanking(["exact-term", "noise"])
    dense = FixedRanking(["semantic-synonym", "noise"])
    texts = [hit.text for hit in HybridRetriever([lexical, dense]).search("q", k=3)]
    assert "exact-term" in texts
    assert "semantic-synonym" in texts


def test_score_follows_the_rrf_formula():
    """First place in both lists sums to 2 / (k + 1)."""
    fused = HybridRetriever([FixedRanking(["a"]), FixedRanking(["a"])])
    assert fused.search("q", k=1)[0].score == pytest.approx(2 / (RRF_K + 1))


def test_results_come_back_in_descending_order():
    fused = HybridRetriever([FixedRanking(["a", "b", "c"]), FixedRanking(["c", "b", "a"])])
    scores = [hit.score for hit in fused.search("q", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_no_duplicates_across_lists():
    fused = HybridRetriever([FixedRanking(["a", "b"]), FixedRanking(["a", "b"])])
    texts = [hit.text for hit in fused.search("q", k=5)]
    assert len(texts) == len(set(texts))


def test_indexes_every_member():
    lexical, dense = FixedRanking(["a"]), FixedRanking(["a"])
    HybridRetriever([lexical, dense]).index(["doc1", "doc2"])
    assert lexical.indexed == ["doc1", "doc2"] == dense.indexed


def test_works_with_a_single_retriever():
    hits = HybridRetriever([FixedRanking(["a", "b"])]).search("q", k=2)
    assert [hit.text for hit in hits] == ["a", "b"]


def test_requires_at_least_one_retriever():
    with pytest.raises(ValueError, match="at least one retriever"):
        HybridRetriever([])


def test_a_larger_constant_flattens_the_ranking():
    members = [FixedRanking(["a", "b"]), FixedRanking(["b", "a"])]
    hits = HybridRetriever(members, rrf_k=1000).search("q", k=2)
    assert abs(hits[0].score - hits[1].score) < 1e-4


def test_state_round_trip_covers_every_member():
    original = HybridRetriever([FixedRanking(["a"]), FixedRanking(["b"])])
    restored = HybridRetriever([FixedRanking([]), FixedRanking([])])
    restored.load_state_dict(original.state_dict())
    assert [member.texts for member in restored.retrievers] == [["a"], ["b"]]
