"""HTTP interface to the RAG agent.

The index is built once, at startup, and reused for every request: indexing is
expensive and the corpus does not change while the service is alive.

    uvicorn rag_agent.api:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_settings
from .factory import build_retriever
from .pipeline import RagPipeline

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the index on startup and release it on shutdown."""
    settings = get_settings()
    documents = sorted(path for path in settings.docs_dir.rglob("*.txt") if path.is_file())
    if not documents:
        raise RuntimeError(f"No .txt documents found in: {settings.docs_dir}")

    pipeline = RagPipeline(
        chunk_size=settings.chunk_size,
        overlap=settings.overlap,
        k=settings.top_k,
        backend=settings.backend,
        retriever=build_retriever(
            settings.backend,
            model_name=settings.embedding_model,
            rerank=settings.rerank,
            reranker_model=settings.reranker_model,
        ),
        cache_dir=settings.cache_dir,
    ).index_paths(documents)

    _state.update(
        pipeline=pipeline,
        documents=len(documents),
        backend=settings.backend,
        rerank=settings.rerank,
        cached=pipeline.loaded_from_cache,
    )
    yield
    _state.clear()


app = FastAPI(
    title="rag-agent-lab",
    version="0.5.0",
    summary="Ask questions in natural language about your own documents.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    """Body of a query to the agent."""

    question: str = Field(min_length=1, max_length=1000, examples=["What does RAG stand for?"])
    k: int = Field(default=3, ge=1, le=20, description="Passages to retrieve")


class Passage(BaseModel):
    """A retrieved passage with its relevance score."""

    text: str
    score: float


class AskResponse(BaseModel):
    """The agent's answer together with the context supporting it."""

    answer: str
    passages: list[Passage]


@app.get("/health", summary="Service status")
def health() -> dict[str, Any]:
    """Report whether the index is loaded and how it was built."""
    return {
        "status": "ok",
        "documents": _state.get("documents", 0),
        "backend": _state.get("backend", "tfidf"),
        "rerank": _state.get("rerank", False),
        "loaded_from_cache": _state.get("cached", False),
    }


@app.post("/ask", response_model=AskResponse, summary="Ask the agent")
def ask(request: AskRequest) -> AskResponse:
    """Retrieve the relevant passages and compose an answer from them."""
    pipeline: RagPipeline | None = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="The index is not available yet.")

    hits = pipeline.retriever.search(request.question, k=request.k)
    return AskResponse(
        answer=pipeline.answer_from(request.question, hits),
        passages=[Passage(text=hit.text, score=hit.score) for hit in hits],
    )
