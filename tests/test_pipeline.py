"""Tests del pipeline completo: de archivos en disco a respuesta."""

import pytest

from rag_agent.pipeline import RagPipeline


@pytest.fixture
def corpus(tmp_path):
    """Un corpus mínimo de dos documentos claramente distintos."""
    (tmp_path / "rag.txt").write_text(
        "RAG combina recuperación de documentos con generación de texto.",
        encoding="utf-8",
    )
    (tmp_path / "cocina.txt").write_text(
        "La paella valenciana lleva arroz, azafrán y garrofón.",
        encoding="utf-8",
    )
    return sorted(tmp_path.glob("*.txt"))


def test_recupera_el_documento_del_tema_preguntado(corpus):
    pipeline = RagPipeline(k=1).index_paths(corpus)
    hits = pipeline.retriever.search("¿Qué lleva la paella?", k=1)
    assert "paella" in hits[0].text


def test_ask_devuelve_una_respuesta_no_vacia(corpus):
    respuesta = RagPipeline(k=1).index_paths(corpus).ask("¿Qué es RAG?")
    assert respuesta.strip()
    assert "recuperación" in respuesta


def test_answer_from_no_repite_la_busqueda(corpus):
    """answer_from debe usar los fragmentos que se le pasan, sin buscar de nuevo."""
    pipeline = RagPipeline().index_paths(corpus)
    hits = pipeline.retriever.search("paella", k=1)
    assert "paella" in pipeline.answer_from("paella", hits)


def test_indexar_sin_documentos_falla(tmp_path):
    with pytest.raises(ValueError):
        RagPipeline().index_paths([])
