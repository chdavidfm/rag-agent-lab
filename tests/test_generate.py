"""Tests de la capa de generación, en sus dos modos.

El modo local debe ser siempre extractivo. El modo LLM se prueba con un
cliente falso: se verifica el contrato (qué prompt se envía, qué modelo se
pide, qué se devuelve) sin gastar una sola llamada real.
"""

import pytest

from rag_agent import generate
from rag_agent.generate import PROMPT, answer


@pytest.fixture(autouse=True)
def sin_credenciales(monkeypatch):
    """Modo local por defecto, aunque el entorno tenga una clave."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_sin_contexto_lo_dice_en_lugar_de_inventar():
    assert "No encontré" in answer("¿Qué es RAG?", [])


def test_la_respuesta_local_solo_contiene_los_fragmentos_dados():
    fragmentos = ["El cielo es azul.", "El agua hierve a 100 grados."]
    respuesta = answer("¿De qué color es el cielo?", fragmentos)
    for fragmento in fragmentos:
        assert fragmento in respuesta


# --- Modo LLM ------------------------------------------------------------


class FakeCompletions:
    def __init__(self, registro):
        self._registro = registro

    def create(self, **kwargs):
        self._registro.update(kwargs)

        class Mensaje:
            content = "Respuesta redactada por el modelo."

        class Eleccion:
            message = Mensaje()

        class Respuesta:
            choices = [Eleccion()]

        return Respuesta()


class FakeOpenAI:
    """Doble del cliente de OpenAI que registra cómo se le llamó."""

    registro: dict = {}

    def __init__(self, **kwargs):
        FakeOpenAI.registro = {"init": kwargs}
        self.chat = type("Chat", (), {"completions": FakeCompletions(FakeOpenAI.registro)})()


@pytest.fixture
def openai_falso(monkeypatch):
    """Sustituye el import perezoso de 'openai' por el doble."""
    modulo = type("ModuloFalso", (), {"OpenAI": FakeOpenAI})
    monkeypatch.setitem(__import__("sys").modules, "openai", modulo)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
    return FakeOpenAI


def test_con_credenciales_se_usa_el_llm(openai_falso):
    assert answer("¿Qué es RAG?", ["contexto"]) == "Respuesta redactada por el modelo."


def test_el_prompt_incluye_pregunta_y_contexto(openai_falso, monkeypatch):
    monkeypatch.setenv("RAG_MODEL", "modelo-de-prueba")
    answer("¿Qué es RAG?", ["fragmento uno", "fragmento dos"])

    registro = openai_falso.registro
    enviado = registro["messages"][0]["content"]
    assert "¿Qué es RAG?" in enviado
    assert "fragmento uno" in enviado and "fragmento dos" in enviado
    assert registro["model"] == "modelo-de-prueba"
    assert registro["temperature"] == 0.2


def test_el_prompt_prohibe_inventar():
    assert "no inventes" in PROMPT.lower()


def test_una_respuesta_vacia_del_modelo_no_devuelve_none(openai_falso, monkeypatch):
    class SinContenido(FakeOpenAI):
        def __init__(self, **kwargs):
            class Completions:
                def create(self, **_):
                    class Mensaje:
                        content = None

                    class Eleccion:
                        message = Mensaje()

                    return type("R", (), {"choices": [Eleccion()]})()

            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setitem(
        __import__("sys").modules, "openai", type("M", (), {"OpenAI": SinContenido})
    )
    assert answer("q", ["contexto"]) == ""


def test_la_clave_y_la_url_llegan_al_cliente(openai_falso, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://mi-endpoint.local/v1")
    answer("q", ["contexto"])
    init = openai_falso.registro["init"]
    assert init["api_key"] == "sk-prueba"
    assert init["base_url"] == "https://mi-endpoint.local/v1"


def test_el_modulo_no_importa_openai_al_cargarse():
    """El import debe ser perezoso: sin credenciales no se toca la librería."""
    assert not hasattr(generate, "OpenAI")
