"""Evaluación de la calidad de recuperación.

Sin medir no hay mejora posible: cambiar de TF-IDF a embeddings solo tiene
sentido si se puede demostrar que recupera mejor. Este módulo implementa
las métricas estándar de recuperación de información sobre un conjunto de
preguntas con respuesta conocida.

Métricas
--------
- **Recall@k**: de los fragmentos relevantes que existen, qué proporción
  aparece entre los k recuperados.
- **Hit@k**: en qué proporción de preguntas se recuperó *algo* relevante.
  Responde a "¿el sistema tenía la información delante?".
- **MRR** (Mean Reciprocal Rank): 1/posición del primer acierto. Premia
  colocar lo relevante arriba, no solo incluirlo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .retriever import Hit, Retriever


@dataclass(frozen=True)
class EvalCase:
    """Una pregunta con las claves que debe contener un fragmento relevante."""

    question: str
    expected: list[str]

    def is_relevant(self, hit: Hit) -> bool:
        """Un fragmento es relevante si contiene alguna clave esperada."""
        text = hit.text.lower()
        return any(key.lower() in text for key in self.expected)


@dataclass(frozen=True)
class CaseResult:
    """Resultado de evaluar una pregunta concreta."""

    case: EvalCase
    rank: int | None
    relevant_found: int

    @property
    def hit(self) -> bool:
        """Se recuperó al menos un fragmento relevante."""
        return self.rank is not None

    @property
    def reciprocal_rank(self) -> float:
        """1/posición del primer acierto; 0 si no hubo ninguno."""
        return 1.0 / self.rank if self.rank else 0.0


@dataclass(frozen=True)
class EvalReport:
    """Métricas agregadas de una evaluación completa."""

    k: int
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def hit_rate(self) -> float:
        """Proporción de preguntas con al menos un acierto (Hit@k)."""
        if not self.results:
            return 0.0
        return sum(result.hit for result in self.results) / self.total

    @property
    def mrr(self) -> float:
        """Media de los rangos recíprocos (MRR)."""
        if not self.results:
            return 0.0
        return sum(result.reciprocal_rank for result in self.results) / self.total

    @property
    def recall(self) -> float:
        """Recall@k medio: relevantes recuperados sobre relevantes esperados."""
        if not self.results:
            return 0.0
        ratios = [
            min(result.relevant_found / len(result.case.expected), 1.0)
            for result in self.results
            if result.case.expected
        ]
        return sum(ratios) / len(ratios) if ratios else 0.0

    def as_dict(self) -> dict[str, float | int]:
        """Métricas en un diccionario, listo para serializar o comparar."""
        return {
            "k": self.k,
            "casos": self.total,
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "recall": round(self.recall, 4),
        }


def load_cases(path: Path) -> list[EvalCase]:
    """Carga casos desde un archivo JSONL (una pregunta por línea)."""
    cases: list[EvalCase] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
            cases.append(EvalCase(question=raw["question"], expected=list(raw["expected"])))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}: línea {number} mal formada ({exc})") from exc
    if not cases:
        raise ValueError(f"{path}: no contiene ningún caso de evaluación")
    return cases


def evaluate(retriever: Retriever, cases: list[EvalCase], k: int = 3) -> EvalReport:
    """Evalúa el recuperador sobre los casos dados."""
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
