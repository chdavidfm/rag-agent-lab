"""Composing the final answer from the retrieved passages.

Two modes, so the project works from the very first run:

* **Without credentials** the answer is extractive: the most relevant passages
  are returned verbatim. Nothing is invented, and nothing is downloaded.
* **With OPENAI_API_KEY** a language model writes the answer, grounded strictly
  in the retrieved context.
"""

from __future__ import annotations

from .config import get_settings

PROMPT = """You answer strictly from the provided context.
If the answer is not in the context, say so plainly. Never invent information.

Context:
{context}

Question: {question}
Answer:"""


def answer(question: str, context_chunks: list[str]) -> str:
    """Answer ``question`` using ``context_chunks`` as the only source."""
    if not context_chunks:
        return "No relevant information was found in the documents."
    settings = get_settings()
    if settings.llm_enabled:
        return _llm_answer(question, "\n\n---\n\n".join(context_chunks))
    return _extractive_answer(context_chunks)


def _extractive_answer(context_chunks: list[str]) -> str:
    """Return the passages verbatim, with no interpretation added."""
    joined = "\n\n".join(f"• {chunk}" for chunk in context_chunks)
    return "[local mode · no LLM] Most relevant passages:\n\n" + joined


def _llm_answer(question: str, context: str) -> str:
    """Ask the model for an answer confined to the given context."""
    # Imported lazily: 'openai' is only needed when credentials are present.
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.2,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
    )
    return response.choices[0].message.content or ""
