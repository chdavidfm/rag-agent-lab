"""Tests del banco de evaluación.

Las métricas se comprueban contra recuperadores falsos de comportamiento
conocido: así se verifica la fórmula, no el rendimiento del modelo.
"""

import pytest

from rag_agent.evaluation import EvalCase, EvalReport, evaluate, load_cases
from rag_agent.retriever import Hit


class FakeRetriever:
    """Recuperador que devuelve siempre una lista fija de fragmentos."""

    def __init__(self, textos):
        self._textos = textos

    def index(self, docs):
        return self

    def search(self, query, k=3):
        return [Hit(text=t, score=1.0 - i * 0.1) for i, t in enumerate(self._textos[:k])]


CASOS = [EvalCase(question="¿Qué es RAG?", expected=["recuperación"])]


def test_acierto_en_primera_posicion_da_mrr_maximo():
    report = evaluate(FakeRetriever(["habla de recuperación", "ruido"]), CASOS, k=2)
    assert report.hit_rate == 1.0
    assert report.mrr == 1.0


def test_acierto_en_segunda_posicion_da_mrr_un_medio():
    report = evaluate(FakeRetriever(["ruido", "habla de recuperación"]), CASOS, k=2)
    assert report.hit_rate == 1.0
    assert report.mrr == pytest.approx(0.5)


def test_sin_aciertos_todas_las_metricas_a_cero():
    report = evaluate(FakeRetriever(["ruido", "más ruido"]), CASOS, k=2)
    assert report.hit_rate == 0.0
    assert report.mrr == 0.0
    assert report.recall == 0.0


def test_recall_cuenta_los_relevantes_encontrados():
    casos = [EvalCase(question="q", expected=["alfa", "beta"])]
    report = evaluate(FakeRetriever(["texto alfa", "texto beta"]), casos, k=2)
    assert report.recall == 1.0

    parcial = evaluate(FakeRetriever(["texto alfa", "ruido"]), casos, k=2)
    assert parcial.recall == pytest.approx(0.5)


def test_la_relevancia_ignora_mayusculas():
    caso = EvalCase(question="q", expected=["RAG"])
    assert caso.is_relevant(Hit(text="hablamos de rag aquí", score=1.0))


def test_informe_vacio_no_divide_entre_cero():
    vacio = EvalReport(k=3)
    assert vacio.hit_rate == 0.0 and vacio.mrr == 0.0 and vacio.recall == 0.0


def test_carga_de_casos_desde_jsonl(tmp_path):
    archivo = tmp_path / "casos.jsonl"
    archivo.write_text(
        '# comentario\n{"question": "q1", "expected": ["a"]}\n\n'
        '{"question": "q2", "expected": ["b"]}\n',
        encoding="utf-8",
    )
    casos = load_cases(archivo)
    assert [c.question for c in casos] == ["q1", "q2"]


@pytest.mark.parametrize("contenido", ["", "{no es json}\n", '{"question": "falta expected"}\n'])
def test_casos_mal_formados_dan_error_claro(tmp_path, contenido):
    archivo = tmp_path / "malo.jsonl"
    archivo.write_text(contenido, encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(archivo)


def test_el_corpus_de_ejemplo_supera_el_umbral_de_calidad():
    """Prueba de regresión real sobre el corpus y los casos del repositorio."""
    from pathlib import Path

    from rag_agent.pipeline import RagPipeline

    docs = sorted(Path("data/sample").rglob("*.txt"))
    pipeline = RagPipeline(k=3).index_paths(docs)
    report = evaluate(pipeline.retriever, load_cases(Path("data/eval/preguntas.jsonl")), k=3)
    assert report.hit_rate >= 0.8, f"Regresión en recuperación: {report.as_dict()}"
    assert report.mrr >= 0.6, f"Regresión en el orden de resultados: {report.as_dict()}"
