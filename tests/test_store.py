"""Tests for the on-disk index cache.

The cache is an optimisation, so the properties that matter are: it never
serves a stale index, and it never turns a corrupt file into a crash.
"""

import numpy as np
import pytest

from rag_agent.store import IndexCache, fingerprint


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "one.txt").write_text("first document", encoding="utf-8")
    (tmp_path / "two.txt").write_text("second document", encoding="utf-8")
    return sorted(tmp_path.glob("*.txt"))


# --- Fingerprinting -------------------------------------------------------


def test_same_corpus_and_settings_give_the_same_key(corpus):
    assert fingerprint(corpus, backend="tfidf") == fingerprint(corpus, backend="tfidf")


def test_editing_a_document_changes_the_key(corpus):
    before = fingerprint(corpus, backend="tfidf")
    corpus[0].write_text("edited content", encoding="utf-8")
    assert fingerprint(corpus, backend="tfidf") != before


def test_an_edit_of_identical_length_still_changes_the_key(corpus):
    """Hashing content, not size or timestamps, is what makes this safe."""
    original = corpus[0].read_text(encoding="utf-8")
    before = fingerprint(corpus, backend="tfidf")
    corpus[0].write_text("x" * len(original), encoding="utf-8")
    assert fingerprint(corpus, backend="tfidf") != before


def test_changing_a_setting_changes_the_key(corpus):
    assert fingerprint(corpus, backend="tfidf") != fingerprint(corpus, backend="hybrid")
    assert fingerprint(corpus, chunk_size=500) != fingerprint(corpus, chunk_size=200)


def test_the_order_of_settings_is_irrelevant(corpus):
    assert fingerprint(corpus, a=1, b=2) == fingerprint(corpus, b=2, a=1)


# --- Storing and loading --------------------------------------------------


def test_a_missing_entry_is_a_miss(tmp_path):
    assert IndexCache(tmp_path).load("absent") is None


def test_round_trip_preserves_passages(tmp_path):
    cache = IndexCache(tmp_path / "cache")
    cache.save("key", {"docs": ["alpha", "beta"]})
    assert cache.load("key")["docs"] == ["alpha", "beta"]


def test_round_trip_preserves_arrays(tmp_path):
    cache = IndexCache(tmp_path)
    vectors = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    cache.save("key", {"docs": ["a", "b"], "embeddings": vectors})

    restored = cache.load("key")
    assert np.allclose(restored["embeddings"], vectors)
    assert restored["docs"] == ["a", "b"]


def test_round_trip_preserves_nested_structures(tmp_path):
    cache = IndexCache(tmp_path)
    state = {"members": [{"docs": ["a"]}, {"embeddings": np.ones((2, 2), dtype=np.float32)}]}
    cache.save("key", state)

    restored = cache.load("key")
    assert restored["members"][0]["docs"] == ["a"]
    assert np.allclose(restored["members"][1]["embeddings"], np.ones((2, 2)))


def test_a_corrupt_entry_is_a_miss_not_a_crash(tmp_path):
    cache = IndexCache(tmp_path)
    cache.save("key", {"docs": ["a"]})
    (tmp_path / "key.json").write_text("{ this is not json", encoding="utf-8")
    assert cache.load("key") is None


def test_a_missing_array_file_is_a_miss(tmp_path):
    cache = IndexCache(tmp_path)
    cache.save("key", {"embeddings": np.ones((2, 2), dtype=np.float32)})
    (tmp_path / "key.npz").unlink()
    assert cache.load("key") is None


def test_nothing_is_stored_with_pickle(tmp_path):
    """Loading a pickle executes code; a cache file must stay inert."""
    cache = IndexCache(tmp_path)
    cache.save("key", {"docs": ["a"], "embeddings": np.ones((2, 2), dtype=np.float32)})
    for path in tmp_path.iterdir():
        assert b"pickle" not in path.read_bytes().lower()


def test_saving_creates_the_directory(tmp_path):
    cache = IndexCache(tmp_path / "nested" / "cache")
    cache.save("key", {"docs": ["a"]})
    assert cache.load("key") is not None


def test_no_temporary_files_are_left_behind(tmp_path):
    cache = IndexCache(tmp_path)
    cache.save("key", {"docs": ["a"], "embeddings": np.ones((1, 2), dtype=np.float32)})
    assert not list(tmp_path.glob("*.tmp"))
