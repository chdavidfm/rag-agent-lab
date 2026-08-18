"""Retrieval benchmark.

Scores the retriever against questions whose answers are known and, optionally,
fails when it falls below a threshold. That turns quality into a continuous
integration condition: a regression in retrieval breaks the build exactly like
a failing test.

    rag-eval
    rag-eval --backend hybrid --rerank --min-mrr 0.8
    rag-eval --compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_settings
from .evaluation import EvalReport, evaluate, load_cases
from .factory import BACKENDS, build_retriever
from .pipeline import RagPipeline

DEFAULT_CASES = "data/eval/questions.jsonl"


def _measure(docs: list[Path], cases, *, backend: str, rerank: bool, k: int) -> EvalReport:
    """Build the given stack, index the corpus and evaluate it."""
    settings = get_settings()
    retriever = build_retriever(
        backend,
        model_name=settings.embedding_model,
        rerank=rerank,
        reranker_model=settings.reranker_model,
    )
    pipeline = RagPipeline(k=k, backend=backend, retriever=retriever).index_paths(docs)
    return evaluate(pipeline.retriever, cases, k=k)


def _format_report(report: EvalReport, label: str) -> str:
    """Render a single evaluation as a readable block."""
    m = report.as_dict()
    lines = [
        f"{label}   ·   {m['cases']} cases   ·   k = {m['k']}",
        "─" * 56,
        f"  Hit@{m['k']}     {m['hit_rate']:>7.1%}   questions with any relevant passage",
        f"  MRR        {m['mrr']:>7.3f}   how high the first correct answer ranks",
        f"  Recall@{m['k']}  {m['recall']:>7.1%}   coverage of everything relevant",
        "─" * 56,
    ]
    missed = [result for result in report.results if not result.hit]
    if missed:
        lines.append(f"Missed ({len(missed)}):")
        lines += [f"  ✗ {result.case.question}" for result in missed]
    else:
        lines.append("Every question retrieved relevant context.")
    return "\n".join(lines)


def _format_comparison(rows: list[tuple[str, EvalReport]], skipped: list[tuple[str, str]]) -> str:
    """Render the measured configurations side by side, best MRR first."""
    header = f"  {'configuration':<22}{'Hit@k':>9}{'MRR':>9}{'Recall@k':>11}"
    lines = [header, "  " + "─" * (len(header) - 2)]
    for label, report in sorted(rows, key=lambda row: row[1].mrr, reverse=True):
        m = report.as_dict()
        lines.append(f"  {label:<22}{m['hit_rate']:>8.1%}{m['mrr']:>9.3f}{m['recall']:>10.1%}")
    if skipped:
        lines.append("")
        lines.append("  Not measured:")
        lines += [f"    {label} — {reason}" for label, reason in skipped]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark. Returns 0 when thresholds hold, 1 otherwise."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Measure retrieval quality")
    parser.add_argument("--docs", default=str(settings.docs_dir), help="Folder of documents")
    parser.add_argument("--cases", default=DEFAULT_CASES, help="Evaluation cases (JSONL)")
    parser.add_argument("--backend", default=settings.backend, choices=BACKENDS)
    parser.add_argument("--rerank", action="store_true", default=settings.rerank)
    parser.add_argument("--k", type=int, default=settings.top_k, help="Passages to retrieve")
    parser.add_argument("--min-hit-rate", type=float, default=None, help="Minimum Hit@k")
    parser.add_argument("--min-mrr", type=float, default=None, help="Minimum MRR")
    parser.add_argument("--json", action="store_true", help="Emit metrics as JSON")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Evaluate every available configuration and rank them",
    )
    args = parser.parse_args(argv)

    from .cli import collect_documents

    documents = collect_documents(Path(args.docs))
    if not documents:
        print(f"No .txt files found in: {args.docs}", file=sys.stderr)
        return 2
    cases = load_cases(Path(args.cases))

    if args.compare:
        rows: list[tuple[str, EvalReport]] = []
        skipped: list[tuple[str, str]] = []
        for label, backend, rerank in _comparison_matrix():
            try:
                rows.append(
                    (label, _measure(documents, cases, backend=backend, rerank=rerank, k=args.k))
                )
            except ImportError:
                # The optional extra is absent. Report it and keep going: the
                # lexical rows are still worth seeing.
                skipped.append((label, 'requires pip install "rag-agent-lab[embeddings]"'))
        print(_format_comparison(rows, skipped))
        return 0

    report = _measure(documents, cases, backend=args.backend, rerank=args.rerank, k=args.k)
    label = args.backend + (" + rerank" if args.rerank else "")

    if args.json:
        print(json.dumps({"configuration": label, **report.as_dict()}))
    else:
        print(_format_report(report, f"Backend: {label}"))

    unmet = []
    if args.min_hit_rate is not None and report.hit_rate < args.min_hit_rate:
        unmet.append(f"Hit@{args.k} {report.hit_rate:.1%} < {args.min_hit_rate:.1%}")
    if args.min_mrr is not None and report.mrr < args.min_mrr:
        unmet.append(f"MRR {report.mrr:.3f} < {args.min_mrr:.3f}")
    if unmet:
        print("\nThreshold not met:", "; ".join(unmet), file=sys.stderr)
        return 1
    return 0


def _comparison_matrix() -> list[tuple[str, str, bool]]:
    """Configurations compared by --compare, as (label, backend, rerank).

    Only the lexical rows run without downloading a model, so they come first
    and always produce output even on a machine with no network access.
    """
    return [
        ("tfidf", "tfidf", False),
        ("tfidf + rerank", "tfidf", True),
        ("embeddings", "embeddings", False),
        ("embeddings + rerank", "embeddings", True),
        ("hybrid", "hybrid", False),
        ("hybrid + rerank", "hybrid", True),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
