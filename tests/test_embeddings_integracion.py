"""Test de integración del recuperador denso con un modelo real.

Descarga un modelo de Hugging Face (cientos de MB), así que no forma parte
de la batería habitual: se ejecuta bajo demanda con

    pytest -m integracion

Su valor es comprobar lo único que un modelo falso no puede demostrar: que
la búsqueda encuentra por significado cuando la pregunta no comparte
ninguna palabra con el documento.
"""

import pytest

pytest.importorskip("sentence_transformers", reason="requiere el extra [embeddings]")

from rag_agent.embeddings import EmbeddingRetriever  # noqa: E402

pytestmark = pytest.mark.integracion

CORPUS = [
    "El gato duerme en el sofá durante la tarde.",
    "Los aviones despegan del aeropuerto cada hora.",
    "La cocina mediterránea usa mucho aceite de oliva.",
]


@pytest.fixture(scope="module")
def retriever():
    return EmbeddingRetriever().index(CORPUS)


@pytest.mark.parametrize(
    ("consulta", "esperado"),
    [
        ("¿Dónde descansa el felino?", CORPUS[0]),
        ("transporte aéreo", CORPUS[1]),
        ("gastronomía con aceite", CORPUS[2]),
    ],
)
def test_encuentra_por_significado_sin_compartir_palabras(retriever, consulta, esperado):
    assert retriever.search(consulta, k=1)[0].text == esperado


def test_la_similitud_esta_normalizada(retriever):
    for hit in retriever.search("felino", k=3):
        assert -1.0 <= hit.score <= 1.0
