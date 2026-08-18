"""Tests del recuperador denso.

Se inyecta un modelo falso con vectores conocidos: así se prueba la lógica
(normalizado, orden, forma de la salida) sin descargar cientos de MB ni
depender de la calidad de un modelo real.
"""

import numpy as np
import pytest

from rag_agent.embeddings import EmbeddingRetriever


class FakeModel:
    """Modelo con un vector fijo por texto; lo desconocido va a un rincón."""

    VECTORES = {
        "gato": [1.0, 0.0],
        "felino": [0.96, 0.28],  # cercano a "gato": mismo significado
        "avión": [0.0, 1.0],
    }

    def encode(self, texts):
        return np.array([self.VECTORES.get(t, [0.5, 0.5]) for t in texts], dtype=np.float32)


@pytest.fixture
def retriever():
    return EmbeddingRetriever(model=FakeModel()).index(["gato", "avión"])


def test_recupera_por_significado_no_por_palabra(retriever):
    """'felino' no comparte letras con 'gato', pero sí significado."""
    hits = retriever.search("felino", k=1)
    assert hits[0].text == "gato"


def test_devuelve_los_k_pedidos_y_ordenados(retriever):
    hits = retriever.search("felino", k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_las_puntuaciones_son_coseno_normalizado(retriever):
    """Con vectores unitarios, la similitud vive en [-1, 1]."""
    for hit in retriever.search("felino", k=2):
        assert -1.0 <= hit.score <= 1.0


def test_buscar_sin_indexar_falla():
    with pytest.raises(RuntimeError):
        EmbeddingRetriever(model=FakeModel()).search("hola")


def test_indexar_lista_vacia_falla():
    with pytest.raises(ValueError):
        EmbeddingRetriever(model=FakeModel()).index([])


def test_un_vector_nulo_no_provoca_division_entre_cero():
    class ModeloNulo:
        def encode(self, texts):
            return np.zeros((len(texts), 2), dtype=np.float32)

    hits = EmbeddingRetriever(model=ModeloNulo()).index(["a", "b"]).search("c", k=1)
    assert np.isfinite(hits[0].score)
