"""Splitting documents into overlapping passages.

This is the first stage of any RAG system: a long document is cut into pieces
small enough to fit a model's context. The overlap prevents a sentence that
straddles a boundary from losing its surrounding context.

Cuts land on whitespace whenever possible. Slicing at a fixed offset would
split words in half, which reads badly in an answer and leaves the retriever
matching fragments that are not words at all.

Pure Python, no dependencies, so the logic stays easy to follow.
"""

from __future__ import annotations

# How far back a cut may search for whitespace, as a share of the passage
# size. Beyond this the passage would grow too uneven, so a hard cut wins.
_MAX_BACKTRACK = 0.2


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split ``text`` into passages of at most ``chunk_size`` characters.

    Args:
        text: The document to split.
        chunk_size: Maximum length of each passage, in characters.
        overlap: Characters repeated between consecutive passages.

    Returns:
        The non-empty passages, in document order.

    Raises:
        ValueError: If the sizes cannot produce forward progress.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be within [0, chunk_size)")

    text = text.strip()
    if not text:
        return []

    passages: list[str] = []
    start = 0
    while start < len(text):
        end = _cut_point(text, start, chunk_size)
        passage = text[start:end].strip()
        if passage:
            passages.append(passage)
        if end >= len(text):
            break
        start = _resume_point(text, max(end - overlap, start + 1), end)
    return passages


def _resume_point(text: str, candidate: int, ceiling: int) -> int:
    """Return where the next passage should begin.

    The overlap is measured in characters, so it can land inside a word. This
    nudges the start forward to the next whitespace, keeping passages from
    opening on a stray fragment. It never moves past ``ceiling``, which would
    drop text instead of repeating it.
    """
    if candidate <= 0 or text[candidate - 1].isspace():
        return candidate
    boundary = text.find(" ", candidate, ceiling)
    return boundary + 1 if boundary != -1 else candidate


def _cut_point(text: str, start: int, chunk_size: int) -> int:
    """Return where to end a passage starting at ``start``.

    Prefers the last whitespace within the size limit, so words stay intact.
    Falls back to a hard cut when no whitespace is close enough — a run of
    unbroken characters has no better boundary to offer.
    """
    hard_end = start + chunk_size
    if hard_end >= len(text):
        return len(text)

    limit = start + int(chunk_size * (1 - _MAX_BACKTRACK))
    boundary = text.rfind(" ", limit, hard_end)
    return boundary if boundary > start else hard_end
