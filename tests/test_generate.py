"""Tests de la capa de generación en modo local (sin LLM).

El modo local es el que se ejecuta sin credenciales: debe ser siempre
extractivo y no inventar contenido que no esté en el contexto.
"""

import pytest

from rag_agent.generate import answer


@pytest.fixture(autouse=True)
def sin_credenciales(monkeypatch):
    """Garantiza el modo local aunque el entorno tenga una clave configurada."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_sin_contexto_lo_dice_en_lugar_de_inventar():
    assert "No encontré" in answer("¿Qué es RAG?", [])


def test_la_respuesta_solo_contiene_los_fragmentos_dados():
    fragmentos = ["El cielo es azul.", "El agua hierve a 100 grados."]
    respuesta = answer("¿De qué color es el cielo?", fragmentos)
    for fragmento in fragmentos:
        assert fragmento in respuesta
