# 🤖 rag-agent-lab

> Un agente **RAG** (Retrieval-Augmented Generation) mínimo, honesto y didáctico.
> Funciona **100% en local sin claves de API**, y mejora cuando le conectas un LLM.

<p align="left">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white" alt="Tests">
  <img src="https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black" alt="Ruff">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## ✋ Honestidad primero

Estoy aprendiendo IA **construyendo en público**. Este es mi primer laboratorio de
RAG: lo he construido con ayuda de IA mientras entiendo cada pieza, y lo dejo
**deliberadamente simple** para poder explicar *por qué* funciona, no solo *que*
funciona. No es una librería para producción — es una base sólida para aprender
y crecer.

## 🧠 ¿Qué hace?

Le das una carpeta de documentos `.txt` y una pregunta. El sistema:

1. **Trocea** los documentos en fragmentos con solape (`chunk.py`)
2. **Indexa** los fragmentos con TF-IDF (`retriever.py`)
3. **Recupera** los más relevantes para tu pregunta
4. **Genera** una respuesta con ellos (`generate.py`)

```
   documentos.txt
        │
        ▼
   [ trocear ] ──► [ indexar TF-IDF ] ──► [ recuperar top-k ] ──► [ generar ] ──► respuesta
                                                ▲                       │
                                            pregunta ──────────────────┘
```

## 🚀 Cómo probarlo (2 minutos)

```bash
# 1. Clonar e instalar
git clone https://github.com/chdavidfm/rag-agent-lab.git
cd rag-agent-lab
pip install -r requirements.txt

# 2. Preguntar sobre los documentos de ejemplo (¡sin ninguna clave!)
python -m rag_agent.cli --docs data/sample --ask "¿Qué es RAG?"
```

Sin `OPENAI_API_KEY`, responde en **modo local**: te muestra los fragmentos más
relevantes (no inventa nada). Si añades una clave en un archivo `.env`
(ver `.env.example`), un LLM redactará la respuesta apoyándose en esos fragmentos.

## 🧪 Desarrollo

```bash
pip install -r requirements-dev.txt
ruff check .     # estilo
pytest           # tests
```

La integración continua (GitHub Actions) ejecuta lint + tests en cada push.

## 🗺️ Roadmap (lo que quiero aprender construyendo)

- [x] Pipeline RAG end-to-end en local
- [x] Tests y CI en verde
- [ ] Embeddings neuronales (sentence-transformers) como alternativa a TF-IDF
- [ ] Almacén vectorial persistente (FAISS / Chroma)
- [ ] Trocear también PDF y Markdown
- [ ] Pequeña API con FastAPI + interfaz web
- [ ] Evaluación de calidad de las respuestas

## 📚 Lo que voy aprendiendo

Iré anotando aquí las ideas clave que entiendo en cada etapa — para mí y para
quien lea esto empezando como yo.

---

_Construido en público por [David Mejía](https://github.com/chdavidfm) · MIT License_
