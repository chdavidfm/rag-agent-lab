"""API REST del agente RAG.

Expone el pipeline por HTTP para poder consultarlo desde cualquier cliente.
El índice se construye una sola vez, al arrancar, y se reutiliza en cada
petición: indexar es caro y el corpus no cambia mientras el servicio vive.

    uvicorn rag_agent.api:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_settings
from .pipeline import RagPipeline

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Construye el índice al arrancar y lo libera al apagar."""
    settings = get_settings()
    docs = sorted(path for path in settings.docs_dir.rglob("*.txt") if path.is_file())
    if not docs:
        raise RuntimeError(f"No se encontraron documentos .txt en: {settings.docs_dir}")
    _state["pipeline"] = RagPipeline(
        chunk_size=settings.chunk_size,
        overlap=settings.overlap,
        k=settings.top_k,
        backend=settings.backend,
    ).index_paths(docs)
    _state["documents"] = len(docs)
    _state["backend"] = settings.backend
    yield
    _state.clear()


app = FastAPI(
    title="rag-agent-lab",
    version="0.3.0",
    summary="Pregunta en lenguaje natural sobre tus propios documentos.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    """Cuerpo de una consulta al agente."""

    question: str = Field(min_length=1, max_length=1000, examples=["¿Qué es RAG?"])
    k: int = Field(default=3, ge=1, le=20, description="Fragmentos a recuperar")


class Passage(BaseModel):
    """Fragmento recuperado, con su puntuación de relevancia."""

    text: str
    score: float


class AskResponse(BaseModel):
    """Respuesta del agente junto al contexto que la sustenta."""

    answer: str
    passages: list[Passage]


@app.get("/health", summary="Estado del servicio")
def health() -> dict[str, Any]:
    """Comprueba que el índice está cargado y listo para responder."""
    return {
        "status": "ok",
        "documents": _state.get("documents", 0),
        "backend": _state.get("backend", "tfidf"),
    }


@app.post("/ask", response_model=AskResponse, summary="Preguntar al agente")
def ask(request: AskRequest) -> AskResponse:
    """Recupera los fragmentos relevantes y compone la respuesta."""
    pipeline: RagPipeline | None = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="El índice todavía no está disponible.")

    hits = pipeline.retriever.search(request.question, k=request.k)
    answer = pipeline.answer_from(request.question, hits)
    return AskResponse(
        answer=answer,
        passages=[Passage(text=hit.text, score=hit.score) for hit in hits],
    )
