"""Interfaz de línea de comandos del agente RAG.

Ejemplo de uso:

    python -m rag_agent.cli --docs data/sample --ask "¿Qué es RAG?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import RagPipeline


def _collect_docs(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*.txt") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente RAG mínimo y didáctico")
    parser.add_argument("--docs", required=True, help="Carpeta con documentos .txt")
    parser.add_argument("--ask", required=True, help="Pregunta a responder")
    parser.add_argument("--k", type=int, default=3, help="Nº de fragmentos a recuperar")
    args = parser.parse_args()

    docs = _collect_docs(Path(args.docs))
    if not docs:
        raise SystemExit(f"No se encontraron archivos .txt en: {args.docs}")

    pipeline = RagPipeline(k=args.k).index_paths(docs)
    print(pipeline.ask(args.ask))


if __name__ == "__main__":
    main()
