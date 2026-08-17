# rag-agent-lab

> Un agente **RAG** (Retrieval-Augmented Generation) mínimo y didáctico.
> Recupera y responde sobre tus propios documentos: **100 % local, sin claves de API**, y mejora cuando le conectas un LLM.

<p align="left">
  <a href="https://github.com/chdavidfm/rag-agent-lab/actions/workflows/ci.yml"><img src="https://github.com/chdavidfm/rag-agent-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/lint%20%26%20format-ruff-261230?logo=ruff&logoColor=white" alt="Ruff">
  <img src="https://img.shields.io/badge/license-MIT-3da639" alt="MIT License">
</p>

## Qué es

Le das una carpeta de documentos `.txt` y una pregunta; el sistema recupera los
fragmentos más relevantes y construye la respuesta a partir de ellos. Está
escrito para **entenderse de raíz**: cada etapa vive en su propio módulo, con el
mínimo de dependencias y sin cajas negras.

```text
   documentos.txt
        │
        ▼
   trocear  ──►  indexar (TF-IDF)  ──►  recuperar top-k  ──►  generar  ──►  respuesta
                                              ▲                    │
                                          pregunta ────────────────┘
```

| Etapa | Módulo | Responsabilidad |
|-------|--------|-----------------|
| Trocear | `rag_agent/chunk.py` | Partir documentos largos en fragmentos con solape |
| Indexar / Recuperar | `rag_agent/retriever.py` | Vectorizar con TF-IDF y ordenar por similitud coseno |
| Generar | `rag_agent/generate.py` | Respuesta extractiva (local) o redactada por un LLM |
| Orquestar | `rag_agent/pipeline.py` | Unir las etapas en un flujo único |
| CLI | `rag_agent/cli.py` | Interfaz de línea de comandos |

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/chdavidfm/rag-agent-lab.git
cd rag-agent-lab
pip install -e .
```

## Uso

```bash
rag-agent --docs data/sample --ask "¿Qué es RAG?"
```

Sin `OPENAI_API_KEY` el agente responde en **modo local**: devuelve los
fragmentos más relevantes, sin inventar nada. Es rápido, gratuito y offline.

### Modo LLM (opcional)

Para que un modelo redacte la respuesta apoyándose en el contexto recuperado:

```bash
pip install -e ".[llm]"
cp .env.example .env        # y añade tu OPENAI_API_KEY
```

`OPENAI_BASE_URL` permite apuntar a cualquier endpoint compatible con la API de
OpenAI (Ollama, Groq, etc.).

## Desarrollo

```bash
pip install -e ".[dev]"
ruff format --check .   # formato
ruff check .            # lint (E, F, I, UP, B)
pytest                  # tests
```

La integración continua (GitHub Actions) ejecuta estas tres comprobaciones en
cada push y cada pull request.

## Roadmap

- [x] Pipeline RAG completo, en local y sin claves
- [x] Empaquetado instalable, tests y CI en verde
- [ ] Embeddings neuronales (sentence-transformers) como alternativa a TF-IDF
- [ ] Almacén vectorial persistente (FAISS / Chroma)
- [ ] Ingesta de PDF y Markdown además de texto plano
- [ ] API con FastAPI e interfaz web
- [ ] Evaluación automática de la calidad de las respuestas

## Sobre el proyecto

Laboratorio de aprendizaje construido en público por
[David Mejía](https://github.com/chdavidfm) mientras profundiza en IA aplicada.
Deliberadamente pequeño para poder razonar cada decisión; pensado para crecer
etapa a etapa siguiendo el roadmap.

## Licencia

[MIT](LICENSE).
