"""Generar la respuesta final a partir de los fragmentos recuperados.

Dos modos, para que el proyecto funcione desde el minuto cero:

* **Sin clave de API** -> modo extractivo: devuelve los fragmentos más
  relevantes. No inventa nada. Perfecto para aprender y probar offline.
* **Con OPENAI_API_KEY** -> pide a un LLM una respuesta redactada que se
  apoya SOLO en el contexto recuperado.
"""

from __future__ import annotations

import os

PROMPT = """Eres un asistente que responde SOLO con la información del contexto.
Si la respuesta no está en el contexto, dilo con claridad y no inventes.

Contexto:
{context}

Pregunta: {question}
Respuesta:"""


def answer(question: str, context_chunks: list[str]) -> str:
    """Responde a ``question`` usando ``context_chunks`` como única fuente."""
    if os.getenv("OPENAI_API_KEY"):
        context = "\n\n---\n\n".join(context_chunks)
        return _llm_answer(question, context)
    return _extractive_answer(context_chunks)


def _extractive_answer(context_chunks: list[str]) -> str:
    if not context_chunks:
        return "No encontré información relevante en los documentos."
    joined = "\n\n".join(f"• {chunk}" for chunk in context_chunks)
    return "[modo local · sin LLM] Fragmentos más relevantes:\n\n" + joined


def _llm_answer(question: str, context: str) -> str:
    # Import perezoso: solo se necesita 'openai' si hay clave configurada.
    from openai import OpenAI

    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
    model = os.getenv("RAG_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(context=context, question=question),
            }
        ],
    )
    return response.choices[0].message.content or ""
