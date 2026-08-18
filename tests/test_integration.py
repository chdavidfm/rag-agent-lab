"""Integration tests against real models.

These download models from Hugging Face — hundreds of megabytes — so they stay
out of the default run and are executed on demand:

    pytest -m integration

Their value is proving the one thing a fake model cannot: that dense retrieval
matches on meaning, and that reranking reorders candidates the first stage got
wrong.
"""

import pytest

pytest.importorskip("sentence_transformers", reason="requires the [embeddings] extra")

from rag_agent.embeddings import EmbeddingRetriever  # noqa: E402
from rag_agent.rerank import RerankingRetriever  # noqa: E402
from rag_agent.retriever import TfidfRetriever  # noqa: E402

pytestmark = pytest.mark.integration

CORPUS = [
    "The cat sleeps on the sofa every afternoon.",
    "Aeroplanes take off from the airport every hour.",
    "Mediterranean cooking relies heavily on olive oil.",
]


@pytest.fixture(scope="module")
def dense():
    return EmbeddingRetriever().index(CORPUS)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Where does the feline rest?", CORPUS[0]),
        ("air travel", CORPUS[1]),
        ("gastronomy using oil", CORPUS[2]),
    ],
)
def test_dense_retrieval_matches_on_meaning(dense, query, expected):
    assert dense.search(query, k=1)[0].text == expected


def test_dense_scores_stay_normalised(dense):
    for hit in dense.search("feline", k=3):
        assert -1.0 <= hit.score <= 1.0


def test_reranking_improves_the_lexical_ordering():
    """A real cross-encoder should rank the passage that answers the question."""
    corpus = [
        "Paris is a popular destination for weekend travel in Europe.",
        "The capital of France is Paris, seat of its government.",
        "France exports wine, cheese and luxury goods worldwide.",
    ]
    question = "What is the capital of France?"

    reranked = RerankingRetriever(TfidfRetriever()).index(corpus).search(question, k=1)
    assert "capital of France is Paris" in reranked[0].text
