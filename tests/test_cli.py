"""Tests for both command-line entry points.

They are invoked exactly as a shell would, checking output and exit codes: a
non-zero code is what makes a script or a CI job fail.
"""

import json

import pytest

from rag_agent import cli, eval_cli


@pytest.fixture(autouse=True)
def local_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RAG_RERANK", raising=False)
    # Keep every run's cache inside the temporary directory.
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "doc.txt").write_text(
        "RAG combines retrieval of documents with text generation.", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def cases(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"question": "What is RAG?", "expected": ["retrieval"]}\n', encoding="utf-8")
    return path


# --- rag-agent ------------------------------------------------------------


def test_answers_and_exits_successfully(corpus, capsys):
    assert cli.main(["--docs", str(corpus), "--ask", "What is RAG?"]) == 0
    assert "retrieval" in capsys.readouterr().out.lower()


def test_reports_when_there_are_no_documents(tmp_path, capsys):
    assert cli.main(["--docs", str(tmp_path / "empty"), "--ask", "hi"]) == 2
    assert "No .txt files found" in capsys.readouterr().err


def test_rejects_an_unknown_backend(corpus):
    with pytest.raises(SystemExit):
        cli.main(["--docs", str(corpus), "--ask", "hi", "--backend", "magic"])


def test_the_no_cache_flag_still_answers(corpus, capsys):
    assert cli.main(["--docs", str(corpus), "--ask", "What is RAG?", "--no-cache"]) == 0
    assert capsys.readouterr().out.strip()


# --- rag-eval -------------------------------------------------------------


def test_prints_the_metrics(corpus, cases, capsys):
    assert eval_cli.main(["--docs", str(corpus), "--cases", str(cases)]) == 0
    out = capsys.readouterr().out
    assert "Hit@" in out and "MRR" in out and "Recall@" in out


def test_json_output_is_machine_readable(corpus, cases, capsys):
    eval_cli.main(["--docs", str(corpus), "--cases", str(cases), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["hit_rate"] == 1.0
    assert data["configuration"] == "tfidf"


def test_a_missed_threshold_fails_the_run(corpus, tmp_path, capsys):
    impossible = tmp_path / "impossible.jsonl"
    impossible.write_text('{"question": "q", "expected": ["nonexistent"]}\n', encoding="utf-8")

    code = eval_cli.main(
        ["--docs", str(corpus), "--cases", str(impossible), "--min-hit-rate", "0.9"]
    )
    assert code == 1
    assert "Threshold not met" in capsys.readouterr().err


def test_a_met_threshold_passes(corpus, cases):
    assert eval_cli.main(["--docs", str(corpus), "--cases", str(cases), "--min-mrr", "0.5"]) == 0


def test_reports_when_there_are_no_documents_to_evaluate(tmp_path, cases):
    assert eval_cli.main(["--docs", str(tmp_path / "empty"), "--cases", str(cases)]) == 2


def test_the_report_lists_the_questions_that_missed(corpus, tmp_path, capsys):
    missing = tmp_path / "missing.jsonl"
    missing.write_text('{"question": "unanswerable?", "expected": ["absent"]}\n', encoding="utf-8")
    eval_cli.main(["--docs", str(corpus), "--cases", str(missing)])
    assert "unanswerable?" in capsys.readouterr().out


def test_compare_degrades_gracefully_without_optional_extras(corpus, cases, capsys, monkeypatch):
    """Missing extras must be reported, not crash the comparison."""
    import rag_agent.eval_cli as module

    real_measure = module._measure

    def only_lexical(*args, backend, rerank, **kwargs):
        if backend != "tfidf" or rerank:
            raise ImportError("sentence-transformers is not installed")
        return real_measure(*args, backend=backend, rerank=rerank, **kwargs)

    monkeypatch.setattr(module, "_measure", only_lexical)

    assert module.main(["--docs", str(corpus), "--cases", str(cases), "--compare"]) == 0
    out = capsys.readouterr().out
    assert "tfidf" in out
    assert "Not measured:" in out
    assert "rag-agent-lab[embeddings]" in out


def test_compare_ranks_configurations_by_mrr(corpus, cases, capsys):
    assert eval_cli.main(["--docs", str(corpus), "--cases", str(cases), "--compare"]) == 0
    assert "configuration" in capsys.readouterr().out
