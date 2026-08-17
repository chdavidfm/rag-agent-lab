"""Test del recuperador: debe poner primero el fragmento más relevante."""

from rag_agent.retriever import TfidfRetriever


def test_recupera_el_documento_relevante_primero():
    docs = [
        "Los gatos son mamíferos felinos domésticos.",
        "Python es un lenguaje de programación muy popular.",
        "RAG combina recuperación de información con generación de texto.",
    ]
    retriever = TfidfRetriever().index(docs)
    hits = retriever.search("¿Qué es RAG?", k=1)
    assert len(hits) == 1
    assert "RAG" in hits[0].text


def test_search_sin_index_falla():
    import pytest

    with pytest.raises(RuntimeError):
        TfidfRetriever().search("hola")
