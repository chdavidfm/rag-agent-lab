"""Tests de la fusión híbrida (Reciprocal Rank Fusion).

Se combinan recuperadores falsos con rankings conocidos: así se comprueba
la fórmula de fusión, no la calidad de un modelo concreto.
"""

import pytest

from rag_agent.hybrid import RRF_K, HybridRetriever
from rag_agent.retriever import Hit


class RankingFijo:
    """Recuperador que siempre devuelve el mismo orden de documentos."""

    def __init__(self, textos):
        self.textos = textos
        self.indexado = None

    def index(self, docs):
        self.indexado = docs
        return self

    def search(self, query, k=3):
        return [Hit(text=t, score=1.0 - i * 0.1) for i, t in enumerate(self.textos[:k])]


def test_un_documento_bien_situado_en_ambos_gana():
    """El consenso entre estrategias es la señal más fuerte."""
    lexico = RankingFijo(["consenso", "solo-lexico", "relleno"])
    denso = RankingFijo(["consenso", "solo-denso", "relleno"])
    hits = HybridRetriever([lexico, denso]).search("q", k=3)
    assert hits[0].text == "consenso"


def test_rescata_lo_que_solo_una_estrategia_encuentra():
    """El valor del híbrido: cubrir el punto ciego de la otra estrategia."""
    lexico = RankingFijo(["termino-exacto", "ruido"])
    denso = RankingFijo(["sinonimo-semantico", "ruido"])
    textos = [hit.text for hit in HybridRetriever([lexico, denso]).search("q", k=3)]
    assert "termino-exacto" in textos
    assert "sinonimo-semantico" in textos


def test_la_puntuacion_sigue_la_formula_rrf():
    """Un documento primero en ambas listas suma 2 * 1/(k+1)."""
    retriever = HybridRetriever([RankingFijo(["a"]), RankingFijo(["a"])])
    assert retriever.search("q", k=1)[0].score == pytest.approx(2 / (RRF_K + 1))


def test_los_resultados_salen_ordenados_de_mayor_a_menor():
    lexico = RankingFijo(["a", "b", "c"])
    denso = RankingFijo(["c", "b", "a"])
    scores = [hit.score for hit in HybridRetriever([lexico, denso]).search("q", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_no_devuelve_documentos_repetidos():
    lexico = RankingFijo(["a", "b"])
    denso = RankingFijo(["a", "b"])
    textos = [hit.text for hit in HybridRetriever([lexico, denso]).search("q", k=5)]
    assert len(textos) == len(set(textos))


def test_indexa_el_corpus_en_todos_los_recuperadores():
    lexico, denso = RankingFijo(["a"]), RankingFijo(["a"])
    HybridRetriever([lexico, denso]).index(["doc1", "doc2"])
    assert lexico.indexado == ["doc1", "doc2"]
    assert denso.indexado == ["doc1", "doc2"]


def test_funciona_con_un_solo_recuperador():
    hits = HybridRetriever([RankingFijo(["a", "b"])]).search("q", k=2)
    assert [hit.text for hit in hits] == ["a", "b"]


def test_sin_recuperadores_falla():
    with pytest.raises(ValueError, match="al menos un recuperador"):
        HybridRetriever([])


def test_una_k_mayor_aplana_las_diferencias():
    """Con rrf_k alto, la ventaja de la primera posición se reduce."""
    listas = [RankingFijo(["a", "b"]), RankingFijo(["b", "a"])]
    suave = HybridRetriever(listas, rrf_k=1000).search("q", k=2)
    assert abs(suave[0].score - suave[1].score) < 1e-4
