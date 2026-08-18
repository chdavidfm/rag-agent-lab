"""Tests de las dos interfaces de línea de comandos.

Se invocan como lo haría la terminal, comprobando salida y código de
retorno: un código distinto de cero es lo que hace fallar un script o un
job de integración continua.
"""

import pytest

from rag_agent import cli, eval_cli


@pytest.fixture(autouse=True)
def sin_credenciales(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "doc.txt").write_text(
        "RAG combina recuperación de documentos con generación de texto.",
        encoding="utf-8",
    )
    return tmp_path


def test_cli_responde_y_termina_con_exito(corpus, capsys):
    codigo = cli.main(["--docs", str(corpus), "--ask", "¿Qué es RAG?"])
    assert codigo == 0
    assert "recuperación" in capsys.readouterr().out


def test_cli_avisa_si_no_hay_documentos(tmp_path, capsys):
    codigo = cli.main(["--docs", str(tmp_path), "--ask", "hola"])
    assert codigo == 2
    assert "No se encontraron archivos" in capsys.readouterr().err


def test_cli_rechaza_un_backend_invalido(corpus):
    with pytest.raises(SystemExit):
        cli.main(["--docs", str(corpus), "--ask", "hola", "--backend", "magia"])


# --- Banco de evaluación --------------------------------------------------


@pytest.fixture
def casos(tmp_path):
    archivo = tmp_path / "casos.jsonl"
    archivo.write_text('{"question": "¿Qué es RAG?", "expected": ["recuperación"]}\n', "utf-8")
    return archivo


def test_eval_muestra_las_metricas(corpus, casos, capsys):
    codigo = eval_cli.main(["--docs", str(corpus), "--cases", str(casos)])
    salida = capsys.readouterr().out
    assert codigo == 0
    assert "Hit@" in salida and "MRR" in salida


def test_eval_en_json_es_analizable(corpus, casos, capsys):
    import json

    eval_cli.main(["--docs", str(corpus), "--cases", str(casos), "--json"])
    datos = json.loads(capsys.readouterr().out)
    assert datos["hit_rate"] == 1.0
    assert set(datos) == {"k", "casos", "hit_rate", "mrr", "recall"}


def test_eval_falla_si_no_alcanza_el_umbral(corpus, tmp_path, capsys):
    """Un umbral imposible debe devolver 1 para romper la CI."""
    imposible = tmp_path / "imposible.jsonl"
    imposible.write_text('{"question": "q", "expected": ["inexistente"]}\n', "utf-8")
    codigo = eval_cli.main(
        ["--docs", str(corpus), "--cases", str(imposible), "--min-hit-rate", "0.9"]
    )
    assert codigo == 1
    assert "Umbral no alcanzado" in capsys.readouterr().err


def test_eval_avisa_si_no_hay_documentos(tmp_path, casos, capsys):
    codigo = eval_cli.main(["--docs", str(tmp_path), "--cases", str(casos)])
    assert codigo == 2
