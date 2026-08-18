"""Banco de pruebas de recuperación.

Mide la calidad del recuperador sobre un conjunto de preguntas con
respuesta conocida y, opcionalmente, falla si baja de un umbral. Eso
convierte la calidad en una condición de integración continua: una
regresión en la recuperación rompe el build igual que un test roto.

    rag-eval --docs data/sample --cases data/eval/preguntas.jsonl
    rag-eval --backend embeddings --min-mrr 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_settings
from .evaluation import evaluate, load_cases
from .factory import BACKENDS
from .pipeline import RagPipeline


def _collect_docs(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*.txt") if path.is_file())


def _format_table(report, backend: str) -> str:
    """Compone un resumen legible en la terminal."""
    metrics = report.as_dict()
    lines = [
        f"Backend: {backend}   ·   casos: {metrics['casos']}   ·   k = {metrics['k']}",
        "─" * 52,
        f"  Hit@{metrics['k']}    {metrics['hit_rate']:>7.1%}   preguntas con algún acierto",
        f"  MRR       {metrics['mrr']:>7.3f}   calidad del orden de resultados",
        f"  Recall@{metrics['k']} {metrics['recall']:>7.1%}   cobertura de lo relevante",
        "─" * 52,
    ]
    fallos = [result for result in report.results if not result.hit]
    if fallos:
        lines.append(f"Sin acierto ({len(fallos)}):")
        lines += [f"  ✗ {result.case.question}" for result in fallos]
    else:
        lines.append("Todas las preguntas recuperaron contexto relevante.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la evaluación. Devuelve 0 si pasa los umbrales, 1 si no."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evalúa la calidad de recuperación")
    parser.add_argument("--docs", default=str(settings.docs_dir), help="Carpeta de documentos")
    parser.add_argument(
        "--cases", default="data/eval/preguntas.jsonl", help="Casos de evaluación (JSONL)"
    )
    parser.add_argument("--backend", default=settings.backend, choices=BACKENDS)
    parser.add_argument("--k", type=int, default=settings.top_k, help="Fragmentos a recuperar")
    parser.add_argument("--min-hit-rate", type=float, default=None, help="Hit@k mínimo exigido")
    parser.add_argument("--min-mrr", type=float, default=None, help="MRR mínimo exigido")
    parser.add_argument("--json", action="store_true", help="Salida en JSON")
    args = parser.parse_args(argv)

    docs = _collect_docs(Path(args.docs))
    if not docs:
        print(f"No se encontraron archivos .txt en: {args.docs}", file=sys.stderr)
        return 2

    pipeline = RagPipeline(backend=args.backend, k=args.k).index_paths(docs)
    report = evaluate(pipeline.retriever, load_cases(Path(args.cases)), k=args.k)

    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False))
    else:
        print(_format_table(report, args.backend))

    incumplidos = []
    if args.min_hit_rate is not None and report.hit_rate < args.min_hit_rate:
        incumplidos.append(f"Hit@{args.k} {report.hit_rate:.1%} < {args.min_hit_rate:.1%}")
    if args.min_mrr is not None and report.mrr < args.min_mrr:
        incumplidos.append(f"MRR {report.mrr:.3f} < {args.min_mrr:.3f}")
    if incumplidos:
        print("\nUmbral no alcanzado:", "; ".join(incumplidos), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
