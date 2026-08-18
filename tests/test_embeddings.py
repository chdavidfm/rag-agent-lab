"""Tests for the dense retriever.

A fake encoder with known vectors is injected, so the logic under test is the
normalisation, ordering and shape handling — not the quality of a real model,
and without downloading hundreds of megabytes.
"""

import numpy as np
import pytest

from rag_agent.embeddings import EmbeddingRetriever


class FakeEncoder:
    """Maps known words to fixed vectors; anything else lands in a corner."""

    VECTORS = {
        "cat": [1.0, 0.0],
        "feline": [0.96, 0.28],  # close to "cat": same meaning, different word
        "aeroplane": [0.0, 1.0],
    }

    def encode(self, texts):
        return np.array([self.VECTORS.get(t, [0.5, 0.5]) for t in texts], dtype=np.float32)


@pytest.fixture
def retriever():
    return EmbeddingRetriever(model=FakeEncoder()).index(["cat", "aeroplane"])


def test_matches_on_meaning_rather_than_spelling(retriever):
    """ "feline" shares no characters with "cat", only meaning."""
    assert retriever.search("feline", k=1)[0].text == "cat"


def test_returns_the_requested_number_in_order(retriever):
    hits = retriever.search("feline", k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_scores_are_normalised_cosines(retriever):
    for hit in retriever.search("feline", k=2):
        assert -1.0 <= hit.score <= 1.0


def test_searching_before_indexing_fails():
    with pytest.raises(RuntimeError):
        EmbeddingRetriever(model=FakeEncoder()).search("anything")


def test_indexing_nothing_fails():
    with pytest.raises(ValueError):
        EmbeddingRetriever(model=FakeEncoder()).index([])


def test_a_zero_vector_does_not_divide_by_zero():
    class ZeroEncoder:
        def encode(self, texts):
            return np.zeros((len(texts), 2), dtype=np.float32)

    hit = EmbeddingRetriever(model=ZeroEncoder()).index(["a", "b"]).search("c", k=1)[0]
    assert np.isfinite(hit.score)


def test_state_round_trip_skips_re_encoding(retriever):
    """Restoring must not touch the encoder: that is the point of caching."""

    class ExplodingEncoder:
        def encode(self, texts):
            raise AssertionError("the encoder must not run when restoring")

    restored = EmbeddingRetriever(model=ExplodingEncoder())
    restored.load_state_dict(retriever.state_dict())
    assert restored._docs == ["cat", "aeroplane"]


def test_state_before_indexing_fails():
    with pytest.raises(RuntimeError):
        EmbeddingRetriever(model=FakeEncoder()).state_dict()
