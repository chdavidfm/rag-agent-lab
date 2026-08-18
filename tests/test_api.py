"""Tests de la API REST, ejercitada de extremo a extremo con TestClient."""

import pytest
from fastapi.testclient import TestClient

from rag_agent.api import app


@pytest.fixture(scope="module")
def client():
    """Cliente con el ciclo de vida activo, para que el índice se construya."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_reporta_documentos_indexados(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["documents"] >= 1


def test_ask_devuelve_respuesta_y_fragmentos(client):
    response = client.post("/ask", json={"question": "¿Qué es RAG?", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert len(body["passages"]) == 2
    assert "RAG" in body["passages"][0]["text"]


def test_ask_ordena_los_fragmentos_por_relevancia(client):
    response = client.post("/ask", json={"question": "¿Qué es RAG?", "k": 3})
    scores = [passage["score"] for passage in response.json()["passages"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "hola", "k": 0},
        {"question": "hola", "k": 999},
        {},
    ],
)
def test_ask_rechaza_peticiones_invalidas(client, payload):
    assert client.post("/ask", json=payload).status_code == 422
