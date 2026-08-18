"""Tests for the chunker. Pure functions, so they run instantly."""

import pytest

from rag_agent.chunk import chunk_text


def test_empty_text_yields_no_passages():
    assert chunk_text("") == []
    assert chunk_text("    ") == []


def test_respects_the_size_limit():
    passages = chunk_text("abcdefghij" * 10, chunk_size=30, overlap=5)
    assert len(passages) > 1
    assert all(len(passage) <= 30 for passage in passages)


def test_covers_the_whole_document():
    """With no whitespace to honour, the split is exact and lossless."""
    text = "0123456789" * 5
    assert "".join(chunk_text(text, chunk_size=20, overlap=0)) == text


def test_cuts_land_on_whitespace_so_words_stay_whole():
    text = " ".join(f"word{i:02d}" for i in range(40))
    for passage in chunk_text(text, chunk_size=60, overlap=0):
        for word in passage.split():
            assert word.startswith("word") and len(word) == 6, f"split word: {word!r}"


def test_falls_back_to_a_hard_cut_without_whitespace():
    passages = chunk_text("x" * 100, chunk_size=30, overlap=0)
    assert all(len(passage) <= 30 for passage in passages)
    assert "".join(passages) == "x" * 100


def test_every_word_of_the_document_survives():
    text = " ".join(f"token{i}" for i in range(60))
    covered = " ".join(chunk_text(text, chunk_size=80, overlap=0)).split()
    assert covered == text.split()


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_rejects_sizes_that_cannot_progress(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=size, overlap=overlap)


def test_passages_never_begin_mid_word():
    """The overlap is measured in characters; it must still land on a word."""
    text = " ".join(f"word{i:02d}" for i in range(60))
    for passage in chunk_text(text, chunk_size=70, overlap=20):
        first = passage.split()[0]
        assert first.startswith("word") and len(first) == 6, f"opens mid-word: {first!r}"


def test_overlap_still_repeats_context():
    passages = chunk_text(" ".join(f"w{i:02d}" for i in range(40)), chunk_size=60, overlap=20)
    assert len(passages) > 1
    assert set(passages[0].split()) & set(passages[1].split())
