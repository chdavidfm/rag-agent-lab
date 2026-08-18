"""Tests for answer composition, in both modes.

Local mode must stay strictly extractive. LLM mode is exercised against a fake
client, verifying the contract — which prompt is sent, which model is asked,
what comes back — without spending a single real call.
"""

import sys

import pytest

from rag_agent import generate
from rag_agent.generate import PROMPT, answer


@pytest.fixture(autouse=True)
def without_credentials(monkeypatch):
    """Default to local mode even if the environment carries a key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_without_context_it_says_so_instead_of_inventing():
    assert "No relevant information" in answer("What is RAG?", [])


def test_the_local_answer_contains_only_the_given_passages():
    passages = ["The sky is blue.", "Water boils at 100 degrees."]
    composed = answer("What colour is the sky?", passages)
    for passage in passages:
        assert passage in composed


# --- LLM mode -------------------------------------------------------------


class FakeCompletions:
    def __init__(self, log, content="An answer written by the model."):
        self._log = log
        self._content = content

    def create(self, **kwargs):
        self._log.update(kwargs)
        message = type("Message", (), {"content": self._content})
        choice = type("Choice", (), {"message": message()})
        return type("Response", (), {"choices": [choice()]})()


class FakeOpenAI:
    """Stand-in for the OpenAI client that records how it was called."""

    log: dict = {}
    content = "An answer written by the model."

    def __init__(self, **kwargs):
        type(self).log = {"init": kwargs}
        self.chat = type(
            "Chat", (), {"completions": FakeCompletions(type(self).log, self.content)}
        )()


@pytest.fixture
def fake_openai(monkeypatch):
    """Replace the lazily imported 'openai' module with the double."""
    monkeypatch.setitem(sys.modules, "openai", type("Module", (), {"OpenAI": FakeOpenAI}))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return FakeOpenAI


def test_credentials_switch_to_the_llm(fake_openai):
    assert answer("What is RAG?", ["context"]) == "An answer written by the model."


def test_the_prompt_carries_the_question_and_every_passage(fake_openai, monkeypatch):
    monkeypatch.setenv("RAG_MODEL", "test-model")
    answer("What is RAG?", ["passage one", "passage two"])

    sent = fake_openai.log["messages"][0]["content"]
    assert "What is RAG?" in sent
    assert "passage one" in sent and "passage two" in sent
    assert fake_openai.log["model"] == "test-model"
    assert fake_openai.log["temperature"] == 0.2


def test_the_prompt_forbids_inventing():
    assert "never invent" in PROMPT.lower()


def test_an_empty_completion_does_not_return_none(fake_openai, monkeypatch):
    class Silent(FakeOpenAI):
        content = None

    monkeypatch.setitem(sys.modules, "openai", type("Module", (), {"OpenAI": Silent}))
    assert answer("q", ["context"]) == ""


def test_the_key_and_base_url_reach_the_client(fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://my-endpoint.local/v1")
    answer("q", ["context"])
    assert fake_openai.log["init"]["api_key"] == "sk-test"
    assert fake_openai.log["init"]["base_url"] == "https://my-endpoint.local/v1"


def test_the_module_does_not_import_openai_at_load_time():
    """The import must stay lazy: no credentials, no library touched."""
    assert not hasattr(generate, "OpenAI")
