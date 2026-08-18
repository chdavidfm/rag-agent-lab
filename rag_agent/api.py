"""API REST del agente RAG.

Expone el pipeline por HTTP para poder consultarlo desde cualquier cliente.
El índice se construye una sola vez, al arrancar, y se reutiliza en cada
petición: indexar es caro y el corpus no cambia mientras el servicio vive.

    uvicorn rag_agent.api:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import RagPipeline

# Carpeta de documentos a indexar. Configurable por entorno para que el mismo
# contenedor sirva cualquier corpus sin reconstruir la imagen.
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", "data/sample"))

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Construye el índice al arrancar y lo libera al apagar."""
    docs = sorted(path for path in DOCS_DIR.rglob("*.txt") if path.is_file())
    if not docs:
        raise RuntimeError(f"No se encontraron documentos .txt en: {DOCS_DIR}")
    _state["pipeline"] = RagPipeline().index_paths(docs)
    _state["documents"] = len(docs)
    yield
    _state.clear()


app = FastAPI(
    title="rag-agent-lab",
    version="0.1.0",
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
    return {"status": "ok", "documents": _state.get("documents", 0)}


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
