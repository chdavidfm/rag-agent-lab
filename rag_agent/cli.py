"""Command-line interface for the RAG agent.

rag-agent --ask "What does RAG stand for?"
rag-agent --ask "How are ranked lists combined?" --backend hybrid --rerank
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import get_settings
from .factory import BACKENDS
from .pipeline import RagPipeline


def collect_documents(folder: Path) -> list[Path]:
    """List the .txt files under ``folder``, including nested directories."""
    return sorted(path for path in folder.rglob("*.txt") if path.is_file())


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ask questions about your own documents")
    parser.add_argument("--docs", default=str(settings.docs_dir), help="Folder of .txt documents")
    parser.add_argument("--ask", required=True, help="Question to answer")
    parser.add_argument("--k", type=int, default=settings.top_k, help="Passages to retrieve")
    parser.add_argument(
        "--backend",
        default=settings.backend,
        choices=BACKENDS,
        help="Retrieval strategy: lexical, semantic, or both fused",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=settings.rerank,
        help="Reorder candidates with a cross-encoder for higher precision",
    )
    parser.add_argument("--no-cache", action="store_true", help="Ignore the persisted index")
    args = parser.parse_args(argv)

    documents = collect_documents(Path(args.docs))
    if not documents:
        print(f"No .txt files found in: {args.docs}", file=sys.stderr)
        return 2

    from .factory import build_retriever

    pipeline = RagPipeline(
        chunk_size=settings.chunk_size,
        overlap=settings.overlap,
        k=args.k,
        backend=args.backend,
        retriever=build_retriever(
            args.backend,
            model_name=settings.embedding_model,
            rerank=args.rerank,
            reranker_model=settings.reranker_model,
        ),
        cache_dir=None if args.no_cache else settings.cache_dir,
    ).index_paths(documents)

    print(pipeline.ask(args.ask))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
