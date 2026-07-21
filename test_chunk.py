"""Tests del troceador. Son puros (sin dependencias externas), así que corren
rápido y siempre en CI."""

import pytest

from rag_agent.chunk import chunk_text


def test_texto_vacio_devuelve_lista_vacia():
    assert chunk_text("") == []
    assert chunk_text("    ") == []


def test_respeta_tamano_y_produce_varios_fragmentos():
    text = "abcdefghij" * 10  # 100 caracteres
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    assert len(chunks) > 1
    assert all(len(chunk) <= 30 for chunk in chunks)


def test_cubre_todo_el_texto():
    text = "0123456789" * 5  # 50 caracteres
    chunks = chunk_text(text, chunk_size=20, overlap=0)
    assert "".join(chunks) == text


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        chunk_text("hola", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("hola", chunk_size=10, overlap=10)
