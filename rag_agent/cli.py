"""Interfaz de línea de comandos del agente RAG.

rag-agent --docs data/sample --ask "¿Qué es RAG?"
rag-agent --ask "¿Qué es RAG?" --backend embeddings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import get_settings
from .factory import BACKENDS
from .pipeline import RagPipeline


def _collect_docs(folder: Path) -> list[Path]:
    """Lista los .txt de la carpeta, incluidas subcarpetas."""
    return sorted(path for path in folder.rglob("*.txt") if path.is_file())


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada. Devuelve el código de salida del proceso."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Agente RAG mínimo y didáctico")
    parser.add_argument(
        "--docs", default=str(settings.docs_dir), help="Carpeta con documentos .txt"
    )
    parser.add_argument("--ask", required=True, help="Pregunta a responder")
    parser.add_argument("--k", type=int, default=settings.top_k, help="Fragmentos a recuperar")
    parser.add_argument(
        "--backend",
        default=settings.backend,
        choices=BACKENDS,
        help="Estrategia de recuperación: léxica (tfidf) o semántica (embeddings)",
    )
    args = parser.parse_args(argv)

    docs = _collect_docs(Path(args.docs))
    if not docs:
        print(f"No se encontraron archivos .txt en: {args.docs}", file=sys.stderr)
        return 2

    pipeline = RagPipeline(
        chunk_size=settings.chunk_size,
        overlap=settings.overlap,
        k=args.k,
        backend=args.backend,
    ).index_paths(docs)
    print(pipeline.ask(args.ask))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
