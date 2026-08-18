"""Tests for the evaluation harness.

Metrics are checked against retrievers with known behaviour, so what is
verified is the formula rather than the performance of any model. The final
test is a genuine regression guard over the corpus in the repository.
"""

from pathlib import Path

import pytest

from rag_agent.evaluation import EvalCase, EvalReport, evaluate, load_cases
from rag_agent.pipeline import RagPipeline
from rag_agent.retriever import Hit


class FixedResults:
    """Returns a fixed list of passages for any query."""

    def __init__(self, texts):
        self._texts = texts

    def index(self, docs):
        return self

    def search(self, query, k=3):
        return [Hit(text=t, score=1.0 - i * 0.1) for i, t in enumerate(self._texts[:k])]


CASES = [EvalCase(question="What is RAG?", expected=["retrieval"])]


def test_a_hit_in_first_place_gives_a_perfect_mrr():
    report = evaluate(FixedResults(["about retrieval", "noise"]), CASES, k=2)
    assert report.hit_rate == 1.0
    assert report.mrr == 1.0


def test_a_hit_in_second_place_halves_the_mrr():
    report = evaluate(FixedResults(["noise", "about retrieval"]), CASES, k=2)
    assert report.hit_rate == 1.0
    assert report.mrr == pytest.approx(0.5)


def test_no_hits_leaves_every_metric_at_zero():
    report = evaluate(FixedResults(["noise", "more noise"]), CASES, k=2)
    assert (report.hit_rate, report.mrr, report.recall) == (0.0, 0.0, 0.0)


def test_recall_counts_how_many_expected_keys_were_found():
    cases = [EvalCase(question="q", expected=["alpha", "beta"])]
    assert evaluate(FixedResults(["text alpha", "text beta"]), cases, k=2).recall == 1.0
    assert evaluate(FixedResults(["text alpha", "noise"]), cases, k=2).recall == pytest.approx(0.5)


def test_relevance_ignores_letter_case():
    assert EvalCase(question="q", expected=["RAG"]).is_relevant(Hit(text="about rag", score=1.0))


def test_an_empty_report_does_not_divide_by_zero():
    empty = EvalReport(k=3)
    assert (empty.hit_rate, empty.mrr, empty.recall, empty.total) == (0.0, 0.0, 0.0, 0)


def test_metrics_serialise_with_stable_keys():
    report = evaluate(FixedResults(["about retrieval"]), CASES, k=1)
    assert set(report.as_dict()) == {"k", "cases", "hit_rate", "mrr", "recall"}


# --- Loading cases --------------------------------------------------------


def test_loads_cases_skipping_blanks_and_comments(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '# a comment\n{"question": "q1", "expected": ["a"]}\n\n'
        '{"question": "q2", "expected": ["b"]}\n',
        encoding="utf-8",
    )
    assert [case.question for case in load_cases(path)] == ["q1", "q2"]


@pytest.mark.parametrize(
    "content", ["", "{not json}\n", '{"question": "missing expected"}\n', "[]\n"]
)
def test_malformed_files_raise_a_clear_error(tmp_path, content):
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path)


def test_the_error_names_the_offending_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "ok", "expected": ["a"]}\n{oops}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_cases(path)


# --- Regression guard over the repository corpus --------------------------


def test_the_shipped_corpus_meets_the_quality_bar():
    """Runs the real corpus and the real questions, as CI does."""
    documents = sorted(Path("data/sample").rglob("*.txt"))
    pipeline = RagPipeline(k=3).index_paths(documents)
    report = evaluate(pipeline.retriever, load_cases(Path("data/eval/questions.jsonl")), k=3)

    assert report.total == 12, "the golden set should not shrink unnoticed"
    assert report.hit_rate >= 0.85, f"retrieval regression: {report.as_dict()}"
    assert report.mrr >= 0.70, f"ranking regression: {report.as_dict()}"
