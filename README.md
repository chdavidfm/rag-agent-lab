# rag-agent-lab

> Un agente **RAG** (Retrieval-Augmented Generation) mínimo y didáctico.
> Pregunta en lenguaje natural sobre tus propios documentos: **100 % local, sin claves de API**, con CLI, API REST y Docker.

<p align="left">
  <a href="https://github.com/chdavidfm/rag-agent-lab/actions/workflows/ci.yml"><img src="https://github.com/chdavidfm/rag-agent-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-3776AB?logo=python&logoColor=white" alt="Python 3.10–3.12">
  <img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/lint%20%26%20format-ruff-261230?logo=ruff&logoColor=white" alt="Ruff">
  <img src="https://img.shields.io/badge/license-MIT-3da639" alt="MIT License">
</p>

## Qué es

Le das una carpeta de documentos y una pregunta; el sistema recupera los
fragmentos más relevantes y construye la respuesta a partir de ellos —nunca
inventando fuera de ese contexto—. Está escrito para **entenderse de raíz**:
cada etapa vive en su propio módulo, con el mínimo de dependencias y sin cajas
negras.

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
| Recuperación densa | `rag_agent/embeddings.py` | Búsqueda por significado con embeddings |
| Selección de estrategia | `rag_agent/factory.py` | Construye el recuperador elegido |
| Evaluación | `rag_agent/evaluation.py` | Métricas Hit@k, MRR y Recall@k |
| Configuración | `rag_agent/config.py` | Ajustes desde el entorno y `.env` |
| CLI | `rag_agent/cli.py` | Interfaz de línea de comandos |
| API | `rag_agent/api.py` | Servicio HTTP con FastAPI |

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/chdavidfm/rag-agent-lab.git
cd rag-agent-lab
pip install -e .
```

## Uso desde la terminal

```bash
rag-agent --docs data/sample --ask "¿Qué es RAG?"
```

Sin `OPENAI_API_KEY` el agente responde en **modo local**: devuelve los
fragmentos más relevantes, sin inventar nada. Es rápido, gratuito y offline.

## Dos formas de buscar

| Backend | Cómo compara | Cuándo conviene |
|---------|--------------|-----------------|
| `tfidf` *(por defecto)* | Palabras compartidas entre pregunta y texto | Instantáneo, sin descargas, sólido con vocabulario común |
| `embeddings` | Significado, mediante vectores neuronales | Encuentra "felino" al preguntar por un gato, aunque la palabra no aparezca |

```bash
pip install -e ".[embeddings]"
rag-agent --ask "¿Dónde descansa el felino?" --backend embeddings
```

El modelo se carga de forma perezosa: quien use el modo léxico nunca
descarga PyTorch.

## Medir la calidad de la recuperación

Cambiar de estrategia solo tiene sentido si se puede demostrar la mejora.
El proyecto incluye un banco de pruebas sobre preguntas con respuesta
conocida (`data/eval/preguntas.jsonl`):

```bash
rag-eval --docs data/sample --cases data/eval/preguntas.jsonl
```

```text
Backend: tfidf   ·   casos: 6   ·   k = 3
────────────────────────────────────────────────────
  Hit@3     100.0%   preguntas con algún acierto
  MRR         0.917   calidad del orden de resultados
  Recall@3   83.3%   cobertura de lo relevante
```

| Métrica | Qué mide |
|---------|----------|
| **Hit@k** | Proporción de preguntas para las que se recuperó algo relevante |
| **MRR** | Posición del primer acierto: premia colocar lo relevante arriba |
| **Recall@k** | Qué parte de todo lo relevante llegó a recuperarse |

Los umbrales convierten la calidad en una condición de integración
continua: si una modificación empeora la recuperación, el build falla.

```bash
rag-eval --min-hit-rate 0.85 --min-mrr 0.75   # devuelve 1 si no se alcanzan
```

## Uso como API

```bash
pip install -e ".[api]"
uvicorn rag_agent.api:app --reload
```

El índice se construye una sola vez al arrancar y se reutiliza en cada
petición. Documentación interactiva en `http://localhost:8000/docs`.

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado del servicio, documentos indexados y backend en uso |
| `POST` | `/ask` | Pregunta al agente; devuelve respuesta y fragmentos usados |

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "¿Qué es RAG?", "k": 3}'
```

```json
{
  "answer": "…",
  "passages": [{ "text": "…", "score": 0.71 }]
}
```

La variable `RAG_DOCS_DIR` define qué carpeta se indexa (por defecto
`data/sample`).

## Docker

```bash
docker build -t rag-agent-lab .
docker run --rm -p 8000:8000 rag-agent-lab
```

La imagen ejecuta el servicio con un usuario sin privilegios e incluye un
`HEALTHCHECK`. Para servir tus propios documentos:

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/mis-docs:/app/docs:ro" -e RAG_DOCS_DIR=/app/docs \
  rag-agent-lab
```

## Modo LLM (opcional)

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
mypy                    # tipos
pytest                  # tests
pytest -m integracion   # tests con modelos reales (descarga)
```

La integración continua ejecuta formato, lint y tipos; la batería de tests
sobre **Python 3.10, 3.11 y 3.12**; la evaluación de calidad con umbrales; y
construye la imagen Docker comprobando que el servicio responde.

## Roadmap

- [x] Pipeline RAG completo, en local y sin claves
- [x] Empaquetado instalable, tests y CI en verde
- [x] API REST con FastAPI y despliegue en Docker
- [x] Embeddings neuronales como alternativa a TF-IDF
- [x] Evaluación con métricas estándar y umbrales en CI
- [ ] Almacén vectorial persistente (FAISS / Chroma)
- [ ] Ingesta de PDF y Markdown además de texto plano
- [ ] Búsqueda híbrida: combinar señal léxica y semántica

## Sobre el proyecto

Laboratorio de aprendizaje construido en público por
[David Mejía](https://github.com/chdavidfm) mientras profundiza en IA aplicada.
Deliberadamente pequeño para poder razonar cada decisión; pensado para crecer
etapa a etapa siguiendo el roadmap.

## Licencia

[MIT](LICENSE).
