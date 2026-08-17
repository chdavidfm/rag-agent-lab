"""Recuperar los fragmentos más relevantes con TF-IDF + similitud coseno.

Funciona 100% en local, sin claves de API ni modelos pesados. Es el corazón
del RAG: dada una pregunta, encontrar qué trozos del corpus se le parecen más.
Más adelante se puede sustituir TF-IDF por embeddings neuronales sin cambiar
el resto del sistema.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Palabras vacías del español ('stopwords'). Aportan poco significado y, si no
# se filtran, ensucian la búsqueda: una pregunta como "¿Qué ES RAG?" podría
# recuperar un documento solo por compartir el "es". Filtrarlas mejora mucho la
# relevancia. (Lección aprendida a base de un test que falló, no en la teoría.)
SPANISH_STOPWORDS = [
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "durante", "e",
    "el", "ella", "ellos", "en", "entre", "era", "es", "esa", "ese", "eso",
    "esta", "estas", "este", "esto", "estos", "hasta", "hay", "la", "las", "le",
    "les", "lo", "los", "más", "me", "mi", "mucho", "muy", "nada", "ni", "no",
    "nos", "o", "otra", "otras", "otro", "otros", "para", "pero", "poco", "por",
    "porque", "que", "qué", "quien", "se", "sin", "sobre", "su", "sus",
    "también", "tanto", "te", "todo", "todos", "tu", "un", "una", "uno", "unos",
    "y", "ya", "yo",
]


@dataclass
class Hit:
    """Un fragmento recuperado junto a su puntuación de relevancia."""

    text: str
    score: float


class TfidfRetriever:
    """Índice sencillo basado en TF-IDF."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words=SPANISH_STOPWORDS)
        self._matrix = None
        self._docs: list[str] = []

    def index(self, docs: list[str]) -> "TfidfRetriever":
        """Construye el índice a partir de una lista de fragmentos."""
        if not docs:
            raise ValueError("No hay documentos que indexar")
        self._docs = docs
        self._matrix = self._vectorizer.fit_transform(docs)
        return self

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Devuelve los ``k`` fragmentos más parecidos a ``query``."""
        if self._matrix is None:
            raise RuntimeError("Debes llamar a index() antes de search()")
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(
            zip(self._docs, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [Hit(text=text, score=float(score)) for text, score in ranked[:k]]
