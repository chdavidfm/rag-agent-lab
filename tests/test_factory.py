"""Tests de la fábrica de recuperadores."""

import pytest

from rag_agent.factory import build_retriever
from rag_agent.retriever import Retriever, TfidfRetriever


def test_construye_el_recuperador_lexico():
    assert isinstance(build_retriever("tfidf"), TfidfRetriever)


def test_acepta_nombres_con_espacios_y_mayusculas():
    assert isinstance(build_retriever("  TFIDF "), TfidfRetriever)


def test_backend_desconocido_da_error_util():
    with pytest.raises(ValueError, match="Backend desconocido"):
        build_retriever("magia")


def test_el_recuperador_cumple_el_contrato():
    """Verifica el Protocol: cualquier backend debe encajar en el pipeline."""
    assert isinstance(build_retriever("tfidf"), Retriever)
