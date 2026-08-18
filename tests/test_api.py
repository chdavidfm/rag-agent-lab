"""Tests for the HTTP interface, exercised end to end with TestClient."""

import pytest
from fastapi.testclient import TestClient

from rag_agent.api import app


@pytest.fixture(scope="module")
def client():
    """A client with the lifespan active, so the index is actually built."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_the_indexed_corpus(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] >= 1
    assert body["backend"] == "tfidf"


def test_health_states_whether_the_index_came_from_cache(client):
    assert isinstance(client.get("/health").json()["loaded_from_cache"], bool)


def test_ask_returns_an_answer_and_its_supporting_passages(client):
    """The defining passage must be retrieved; lexical search does not
    guarantee it ranks first, which is precisely what reranking is for."""
    body = client.post("/ask", json={"question": "What does RAG stand for?", "k": 3}).json()
    assert body["answer"]
    assert len(body["passages"]) == 3
    retrieved = " ".join(passage["text"] for passage in body["passages"])
    assert "Retrieval-Augmented Generation" in retrieved


def test_passages_come_back_ranked(client):
    body = client.post("/ask", json={"question": "What does RAG stand for?", "k": 3}).json()
    scores = [passage["score"] for passage in body["passages"]]
    assert scores == sorted(scores, reverse=True)


def test_k_controls_how_many_passages_are_returned(client):
    body = client.post("/ask", json={"question": "retrieval", "k": 1}).json()
    assert len(body["passages"]) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "hello", "k": 0},
        {"question": "hello", "k": 999},
        {"question": "x" * 1001},
        {},
    ],
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_the_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/ask" in schema["paths"]
    assert schema["info"]["title"] == "rag-agent-lab"
