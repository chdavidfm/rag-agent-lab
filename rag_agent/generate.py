"""Composición de la respuesta a partir de los fragmentos recuperados.

Dos modos, para que el proyecto funcione desde el minuto cero:

* **Sin credenciales** -> modo extractivo: devuelve los fragmentos más
  relevantes. No inventa nada. Rápido, gratuito y offline.
* **Con OPENAI_API_KEY** -> se pide a un LLM una respuesta redactada que se
  apoya únicamente en el contexto recuperado.
"""

from __future__ import annotations

from .config import get_settings

PROMPT = """Eres un asistente que responde SOLO con la información del contexto.
Si la respuesta no está en el contexto, dilo con claridad y no inventes.

Contexto:
{context}

Pregunta: {question}
Respuesta:"""


def answer(question: str, context_chunks: list[str]) -> str:
    """Responde a `question` usando `context_chunks` como única fuente."""
    if not context_chunks:
        return "No encontré información relevante en los documentos."
    settings = get_settings()
    if settings.llm_enabled:
        return _llm_answer(question, "\n\n---\n\n".join(context_chunks))
    return _extractive_answer(context_chunks)


def _extractive_answer(context_chunks: list[str]) -> str:
    """Devuelve los fragmentos tal cual, sin interpretación ni añadidos."""
    joined = "\n\n".join(f"• {chunk}" for chunk in context_chunks)
    return "[modo local · sin LLM] Fragmentos más relevantes:\n\n" + joined


def _llm_answer(question: str, context: str) -> str:
    """Pide al modelo una respuesta ceñida al contexto."""
    # Import perezoso: 'openai' solo hace falta si hay credenciales.
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.2,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
    )
    return response.choices[0].message.content or ""
