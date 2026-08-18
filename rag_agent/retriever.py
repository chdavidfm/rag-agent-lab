"""Recuperación de fragmentos relevantes.

Define el contrato que cumple cualquier recuperador (`Retriever`) y la
implementación léxica basada en TF-IDF. Trabajar contra el contrato —y no
contra una clase concreta— permite cambiar de estrategia (léxica, densa,
híbrida) sin tocar el pipeline ni la API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Palabras vacías del español. Aportan poco significado y, sin filtrarlas,
# ensucian la búsqueda léxica: una pregunta como "¿Qué ES RAG?" podría
# recuperar un documento solo por compartir el "es".
SPANISH_STOPWORDS = [
    "a",
    "al",
    "algo",
    "algunas",
    "algunos",
    "ante",
    "antes",
    "como",
    "con",
    "contra",
    "cual",
    "cuando",
    "de",
    "del",
    "desde",
    "donde",
    "durante",
    "e",
    "el",
    "ella",
    "ellos",
    "en",
    "entre",
    "era",
    "es",
    "esa",
    "ese",
    "eso",
    "esta",
    "estas",
    "este",
    "esto",
    "estos",
    "hasta",
    "hay",
    "la",
    "las",
    "le",
    "les",
    "lo",
    "los",
    "más",
    "me",
    "mi",
    "mucho",
    "muy",
    "nada",
    "ni",
    "no",
    "nos",
    "o",
    "otra",
    "otras",
    "otro",
    "otros",
    "para",
    "pero",
    "poco",
    "por",
    "porque",
    "que",
    "qué",
    "quien",
    "se",
    "sin",
    "sobre",
    "su",
    "sus",
    "también",
    "tanto",
    "te",
    "todo",
    "todos",
    "tu",
    "un",
    "una",
    "uno",
    "unos",
    "y",
    "ya",
    "yo",
]


@dataclass(frozen=True)
class Hit:
    """Un fragmento recuperado junto a su puntuación de relevancia."""

    text: str
    score: float


@runtime_checkable
class Retriever(Protocol):
    """Contrato mínimo de un recuperador.

    Cualquier objeto que sepa indexar una lista de fragmentos y devolver los
    más parecidos a una consulta encaja aquí, sin necesidad de heredar.
    """

    def index(self, docs: list[str]) -> Retriever:
        """Prepara la estructura de búsqueda a partir de los fragmentos."""
        ...

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Devuelve los `k` fragmentos más relevantes, de mayor a menor."""
        ...


class TfidfRetriever:
    """Recuperador léxico: TF-IDF y similitud coseno.

    Rápido, sin modelos que descargar y muy sólido cuando la pregunta
    comparte vocabulario con los documentos. Su límite es que no entiende
    sinónimos: para eso está el recuperador denso.
    """

    name = "tfidf"

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words=SPANISH_STOPWORDS)
        self._matrix = None
        self._docs: list[str] = []

    def index(self, docs: list[str]) -> TfidfRetriever:
        """Construye el índice a partir de una lista de fragmentos."""
        if not docs:
            raise ValueError("No hay documentos que indexar")
        self._docs = docs
        self._matrix = self._vectorizer.fit_transform(docs)
        return self

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Devuelve los `k` fragmentos más parecidos a `query`."""
        if self._matrix is None:
            raise RuntimeError("Debes llamar a index() antes de search()")
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(
            zip(self._docs, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [Hit(text=text, score=float(score)) for text, score in ranked[:k]]
