"""Búsqueda híbrida mediante Reciprocal Rank Fusion (RRF).

Las dos estrategias de recuperación fallan de forma distinta:

* La **léxica** (TF-IDF) acierta con términos exactos —nombres propios,
  códigos, siglas— y falla ante sinónimos.
* La **densa** (embeddings) capta el significado, pero puede pasar por alto
  una coincidencia literal poco frecuente.

Combinarlas cubre el hueco de cada una. El problema es que sus
puntuaciones no son comparables: un 0,7 de coseno no equivale a un 0,7 de
TF-IDF. RRF lo resuelve ignorando las puntuaciones y fusionando solo las
**posiciones**:

    score(d) = Σ  1 / (k + posición del documento d en la lista i)

Un documento bien colocado por ambas estrategias sube; uno que solo una
considera relevante entra igualmente, pero más abajo. Es la técnica que
emplean los motores de búsqueda modernos por ser robusta y no necesitar
calibrar escalas.

Referencia: Cormack, Clarke y Buettcher (SIGIR 2009).
"""

from __future__ import annotations

from .retriever import Hit, Retriever

# Constante de suavizado del artículo original. Amortigua el peso de las
# primeras posiciones para que un único primer puesto no domine la fusión.
RRF_K = 60


class HybridRetriever:
    """Fusiona los resultados de varios recuperadores con RRF."""

    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], rrf_k: int = RRF_K) -> None:
        """Crea el recuperador híbrido.

        Args:
            retrievers: Recuperadores a combinar, ya construidos.
            rrf_k: Constante de suavizado; a mayor valor, menos peso tienen
                las primeras posiciones frente al resto.

        Raises:
            ValueError: Si no se pasa ningún recuperador.
        """
        if not retrievers:
            raise ValueError("La búsqueda híbrida necesita al menos un recuperador")
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def index(self, docs: list[str]) -> HybridRetriever:
        """Indexa el mismo corpus en cada recuperador."""
        for retriever in self.retrievers:
            retriever.index(docs)
        return self

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Devuelve los `k` fragmentos mejor situados tras la fusión.

        Cada recuperador aporta más candidatos de los que se van a devolver:
        un documento puede estar en quinta posición en una lista y en
        primera en la otra, y esa señal se perdería al truncar demasiado
        pronto.
        """
        candidatos = max(k * 3, 10)
        puntuaciones: dict[str, float] = {}

        for retriever in self.retrievers:
            for posicion, hit in enumerate(retriever.search(query, k=candidatos), start=1):
                puntuaciones[hit.text] = puntuaciones.get(hit.text, 0.0) + 1.0 / (
                    self.rrf_k + posicion
                )

        ordenados = sorted(puntuaciones.items(), key=lambda par: par[1], reverse=True)
        return [Hit(text=texto, score=score) for texto, score in ordenados[:k]]
