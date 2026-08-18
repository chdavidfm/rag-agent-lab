"""Measuring retrieval quality.

Nothing improves without measurement: swapping TF-IDF for embeddings is only
worth it if the change can be shown to retrieve better. This module implements
the standard information-retrieval metrics over a set of questions whose
answers are known.

Metrics
-------
- **Recall@k** — of the relevant passages that exist, the share that appears in
  the top k.
- **Hit@k** — the share of questions for which *anything* relevant surfaced.
  It answers "did the model even have the information in front of it?".
- **MRR** — mean reciprocal rank, one over the position of the first correct
  result. It rewards ranking the answer first, not merely including it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .retriever import Hit, Retriever


@dataclass(frozen=True)
class EvalCase:
    """A question and the keys a relevant passage must contain."""

    question: str
    expected: list[str]

    def is_relevant(self, hit: Hit) -> bool:
        """Whether a passage contains any of the expected keys."""
        text = hit.text.lower()
        return any(key.lower() in text for key in self.expected)


@dataclass(frozen=True)
class CaseResult:
    """The outcome of evaluating one question."""

    case: EvalCase
    rank: int | None
    relevant_found: int

    @property
    def hit(self) -> bool:
        """Whether at least one relevant passage was retrieved."""
        return self.rank is not None

    @property
    def reciprocal_rank(self) -> float:
        """One over the position of the first hit; zero when there is none."""
        return 1.0 / self.rank if self.rank else 0.0


@dataclass(frozen=True)
class EvalReport:
    """Aggregated metrics over a full evaluation run."""

    k: int
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Number of questions evaluated."""
        return len(self.results)

    @property
    def hit_rate(self) -> float:
        """Share of questions with at least one relevant passage (Hit@k)."""
        if not self.results:
            return 0.0
        return sum(result.hit for result in self.results) / self.total

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank across every question."""
        if not self.results:
            return 0.0
        return sum(result.reciprocal_rank for result in self.results) / self.total

    @property
    def recall(self) -> float:
        """Mean Recall@k: relevant passages retrieved over those expected."""
        if not self.results:
            return 0.0
        ratios = [
            min(result.relevant_found / len(result.case.expected), 1.0)
            for result in self.results
            if result.case.expected
        ]
        return sum(ratios) / len(ratios) if ratios else 0.0

    def as_dict(self) -> dict[str, float | int]:
        """Return the metrics ready to serialise or compare."""
        return {
            "k": self.k,
            "cases": self.total,
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "recall": round(self.recall, 4),
        }


def load_cases(path: Path) -> list[EvalCase]:
    """Load evaluation cases from a JSONL file, one question per line.

    Raises:
        ValueError: If a line is malformed or the file holds no cases.
    """
    cases: list[EvalCase] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
            cases.append(EvalCase(question=raw["question"], expected=list(raw["expected"])))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}: malformed line {number} ({exc})") from exc
    if not cases:
        raise ValueError(f"{path}: contains no evaluation cases")
    return cases


def evaluate(retriever: Retriever, cases: list[EvalCase], k: int = 3) -> EvalReport:
    """Run every case through ``retriever`` and aggregate the metrics."""
    results: list[CaseResult] = []
    for case in cases:
        hits = retriever.search(case.question, k=k)
        rank: int | None = None
        relevant_found = 0
        for position, hit in enumerate(hits, start=1):
            if case.is_relevant(hit):
                relevant_found += 1
                if rank is None:
                    rank = position
        results.append(CaseResult(case=case, rank=rank, relevant_found=relevant_found))
    return EvalReport(k=k, results=results)
