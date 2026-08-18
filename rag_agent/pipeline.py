"""Orchestration of the full RAG flow.

    ingest -> chunk -> index -> retrieve -> generate

The pipeline holds the stages together and is equally convenient from the CLI,
the API and the evaluation harness. It knows only the `Retriever` contract, so
any retrieval strategy plugs in unchanged.
"""

from __future__ import annotations

from pathlib import Path

from .chunk import chunk_text
from .factory import build_retriever
from .generate import answer
from .retriever import Hit, Retriever
from .store import IndexCache, fingerprint


class RagPipeline:
    """A retrieval-augmented generation pipeline, end to end."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        k: int = 3,
        retriever: Retriever | None = None,
        backend: str = "tfidf",
        cache_dir: Path | None = None,
    ) -> None:
        """Create the pipeline.

        Args:
            chunk_size: Passage length, in characters.
            overlap: Characters shared by consecutive passages.
            k: Passages retrieved per query.
            retriever: An already built retriever. Takes precedence over
                ``backend`` and allows injecting a custom stack or a test
                double.
            backend: Strategy to build when ``retriever`` is not supplied.
            cache_dir: Where to persist the built index. None disables
                caching.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.k = k
        self.backend = backend
        self.retriever = retriever if retriever is not None else build_retriever(backend)
        self.cache = IndexCache(cache_dir) if cache_dir is not None else None
        self.loaded_from_cache = False

    def index_paths(self, paths: list[Path]) -> RagPipeline:
        """Index the given documents, reusing a cached index when valid."""
        key = self._cache_key(paths)

        if self.cache is not None and key is not None:
            cached = self.cache.load(key)
            if cached is not None:
                self.retriever.load_state_dict(cached)
                self.loaded_from_cache = True
                return self

        self.retriever.index(self._read_passages(paths))
        self.loaded_from_cache = False

        if self.cache is not None and key is not None:
            self.cache.save(key, self.retriever.state_dict())
        return self

    def ask(self, question: str) -> str:
        """Retrieve the relevant context and compose an answer from it."""
        return self.answer_from(question, self.retriever.search(question, k=self.k))

    def answer_from(self, question: str, hits: list[Hit]) -> str:
        """Compose an answer from passages that were already retrieved.

        Callers holding the passages — such as the API, which returns them
        alongside the answer — avoid a second search this way.
        """
        return answer(question, [hit.text for hit in hits])

    def _read_passages(self, paths: list[Path]) -> list[str]:
        """Read every document and split it into passages."""
        passages: list[str] = []
        for path in paths:
            try:
                text = Path(path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ValueError(f"Could not read document {path}: {exc}") from exc
            passages.extend(chunk_text(text, self.chunk_size, self.overlap))
        if not passages:
            raise ValueError("The documents contain no text to index")
        return passages

    def _cache_key(self, paths: list[Path]) -> str | None:
        """Fingerprint the corpus and the settings that shape the index."""
        if self.cache is None:
            return None
        try:
            return fingerprint(
                paths,
                backend=self.backend,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                retriever=type(self.retriever).__name__,
            )
        except OSError:
            # An unreadable document is reported later, with context, by
            # _read_passages; here it simply means the cache cannot be used.
            return None
