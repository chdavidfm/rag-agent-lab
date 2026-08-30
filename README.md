# rag-agent-lab

> A compact, transparent **RAG** (Retrieval-Augmented Generation) stack.
> Ask questions in natural language about your own documents — **fully local, no API keys** — with a CLI, an HTTP API and Docker.

<p align="left">
  <a href="https://github.com/chdavidfm/rag-agent-lab/actions/workflows/ci.yml"><img src="https://github.com/chdavidfm/rag-agent-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/chdavidfm/rag-agent-lab/actions/workflows/codeql.yml"><img src="https://github.com/chdavidfm/rag-agent-lab/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"></a>
  <img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-3776AB?logo=python&logoColor=white" alt="Python 3.10–3.12">
  <img src="https://img.shields.io/badge/typed-mypy-1f5082" alt="Typed with mypy">
  <img src="https://img.shields.io/badge/license-MIT-3da639" alt="MIT License">
</p>

## What it is

Point it at a folder of documents and ask a question. The system retrieves the
most relevant passages and builds the answer from them — never inventing
anything outside that context. It is written to be **read end to end**: every
stage lives in its own module, with minimal dependencies and no black boxes.

```text
  documents ──► chunk ──► index ──► retrieve ──► rerank ──► generate ──► answer
                                        ▲          (optional)   │
                                     question ───────────────────────┘
```

| Stage | Module | Responsibility |
|-------|--------|----------------|
| Chunk | `rag_agent/chunk.py` | Split documents into overlapping passages |
| Lexical retrieval | `rag_agent/retriever.py` | TF-IDF vectors ranked by cosine similarity |
| Dense retrieval | `rag_agent/embeddings.py` | Neural embeddings that match on meaning |
| Fusion | `rag_agent/hybrid.py` | Merge rankings with Reciprocal Rank Fusion |
| Reranking | `rag_agent/rerank.py` | Cross-encoder second pass over candidates |
| Persistence | `rag_agent/store.py` | Content-addressed index cache |
| Evaluation | `rag_agent/evaluation.py` | Hit@k, MRR and Recall@k |
| Orchestration | `rag_agent/pipeline.py` | Wire the stages into one flow |
| Interfaces | `rag_agent/cli.py`, `api.py` | Command line and HTTP |

## Install

Requires Python 3.10 or newer.

```bash
pip install "git+https://github.com/chdavidfm/rag-agent-lab@v0.5.0"
```

Every tag is built and published by CI, with a wheel and an sdist attached; the
current one is on the
[releases page](https://github.com/chdavidfm/rag-agent-lab/releases). Each
artefact carries a signed provenance statement, so you can confirm it was built
by this repository's workflow rather than uploaded by someone:

```bash
gh attestation verify rag_agent_lab-0.5.0-py3-none-any.whl \
  --repo chdavidfm/rag-agent-lab
```

To work on the code instead, install it editable from a clone:

```bash
git clone https://github.com/chdavidfm/rag-agent-lab.git
cd rag-agent-lab
pip install -e .
```

## Use it

```bash
rag-agent --ask "What does RAG stand for?"
```

Without `OPENAI_API_KEY` the agent answers in **local mode**: it returns the
most relevant passages verbatim and invents nothing. Fast, free and offline.

## Retrieval strategies

| Backend | How it compares | When it wins |
|---------|-----------------|--------------|
| `tfidf` *(default)* | Words shared between query and passage | Instant, no downloads, strong on exact terms and acronyms |
| `embeddings` | Meaning, through neural vectors | Finds a passage about a feline when you ask about a cat |
| `hybrid` | Fuses both rankings by position | Covers each strategy's blind spot |

```bash
pip install -e ".[embeddings]"
rag-agent --ask "Where does the feline rest?" --backend hybrid --rerank
```

Models load lazily: anyone staying on the lexical backend never downloads
PyTorch.

### Why fusion works

The two strategies produce scores on incompatible scales — a cosine of 0.7 is
nothing like a TF-IDF of 0.7. **Reciprocal Rank Fusion** sidesteps calibration
entirely by discarding the scores and merging only the positions:

```text
score(d) = Σ  1 / (60 + rank of d in list i)
```

A passage ranked highly by both rises to the top; one found by a single
strategy still makes the cut, a little lower. Nothing needs tuning, which is
why modern search engines rely on it (Cormack et al., SIGIR 2009).

### Why reranking helps

First-stage retrieval encodes the query and each passage **separately** — that
is what makes it fast, since passages are embedded once, ahead of time. The
cost is precision: the model never sees the pair together.

A **cross-encoder** reads query and passage jointly and scores their relevance
directly. Far more accurate, far too slow to run over a whole collection. So
production systems use both stages:

```text
query ──► retrieve top 15 (fast) ──► rerank to top 3 (accurate) ──► answer
```

```bash
rag-agent --ask "What is the capital of France?" --rerank
```

## Measuring quality

Switching strategies is only worth it if the improvement can be demonstrated.
The repository ships a golden set of questions with known answers:

```bash
rag-eval
```

```text
Backend: tfidf   ·   12 cases   ·   k = 3
────────────────────────────────────────────────────────
  Hit@3      100.0%   questions with any relevant passage
  MRR         0.875   how high the first correct answer ranks
  Recall@3    75.0%   coverage of everything relevant
────────────────────────────────────────────────────────
```

| Metric | What it captures |
|--------|------------------|
| **Hit@k** | Share of questions where anything relevant surfaced |
| **MRR** | Position of the first correct passage — rewards ranking it first |
| **Recall@k** | Share of all relevant material that was retrieved |

Compare every configuration side by side:

```bash
rag-eval --compare
```

Thresholds turn quality into a CI condition: if a change degrades retrieval,
the build fails.

```bash
rag-eval --min-hit-rate 0.90 --min-mrr 0.80   # exits 1 when unmet
```

A threshold only says whether today is acceptable. Every Monday the measurement
is appended to [`data/quality-history.csv`](data/quality-history.csv) before
the thresholds are applied — including the weeks it falls short, which are the
rows worth having.

## Index persistence

Embedding a corpus is the slowest part of the pipeline and produces identical
vectors whenever the documents are unchanged, so the built index is cached to
disk. A cold start drops from minutes to milliseconds.

The cache is **content-addressed**: its key is a digest of the documents' bytes
and of the settings that shaped the index. Edit a document, change the chunk
size or switch backend, and the key changes — a stale index is never served.

Nothing is written with `pickle`. Loading a pickle executes arbitrary code, and
a cache file is precisely the artefact an attacker would try to replace, so
passages travel as JSON and vectors as a NumPy archive: both inert.

```bash
rag-agent --ask "..." --no-cache    # ignore the persisted index
```

## HTTP API

```bash
pip install -e ".[api]"
uvicorn rag_agent.api:app --reload
```

The index is built once at startup and reused for every request. Interactive
documentation at `http://localhost:8000/docs`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status, corpus size, backend and cache state |
| `POST` | `/ask` | Ask a question; returns the answer and its supporting passages |

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does RAG stand for?", "k": 3}'
```

```json
{
  "answer": "…",
  "passages": [{ "text": "…", "score": 0.71 }]
}
```

## Docker

```bash
docker build -t rag-agent-lab .
docker run --rm -p 8000:8000 rag-agent-lab
```

The image runs as an unprivileged user and ships a `HEALTHCHECK`. To serve your
own documents:

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/my-docs:/app/docs:ro" -e RAG_DOCS_DIR=/app/docs \
  rag-agent-lab
```

## Answer generation with an LLM

To have a model write the answer, grounded in the retrieved context:

```bash
pip install -e ".[llm]"
cp .env.example .env        # then set OPENAI_API_KEY
```

`OPENAI_BASE_URL` points at any OpenAI-compatible endpoint (Ollama, Groq, …).
Every setting is documented in [`.env.example`](.env.example).

## Development

```bash
pip install -e ".[dev]"
ruff format --check .   # formatting
ruff check .            # linting (E, F, I, UP, B)
mypy                    # types
pytest                  # test suite
pytest -m integration   # tests against real models (downloads)
```

CI runs formatting, linting and types; the suite across **Python 3.10, 3.11 and
3.12**; the quality thresholds; and a Docker build that verifies the service
answers.

If an agent is doing the work, [`AGENTS.md`](AGENTS.md) carries the gates and
the traps this project has already fallen into.

The repository also maintains itself:

| Automation | What it does |
|------------|--------------|
| **CodeQL** | Scans for vulnerabilities on every change and weekly |
| **Dependabot** | Opens pull requests with updated dependencies; CI decides if they are safe |
| **Weekly evaluation** | Re-measures quality, records the row, and opens an issue if it dropped |
| **Release** | A tag re-measures retrieval and publishes only if the thresholds hold; artefacts carry signed provenance, and the notes come from `CHANGELOG.md`, so a version without an entry cannot ship |

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and
[SECURITY.md](SECURITY.md) for reporting a vulnerability.

## Roadmap

- [x] End-to-end RAG pipeline, local and key-free
- [x] Installable package, test suite and green CI
- [x] HTTP API and Docker deployment
- [x] Dense retrieval with neural embeddings
- [x] Evaluation with standard metrics and CI thresholds
- [x] Hybrid search through Reciprocal Rank Fusion
- [x] Cross-encoder reranking
- [x] Content-addressed index cache
- [x] Gated releases published from a tag, with signed provenance
- [x] Quality measured weekly and kept as a history
- [ ] Persistent vector store (FAISS / Chroma)
- [ ] PDF and Markdown ingestion
- [ ] Streaming answers over the API

## About

A learning laboratory built in public by
[David Mejía](https://github.com/chdavidfm) while going deep on applied AI.
Deliberately small, so every decision can be reasoned about; designed to grow
one stage at a time.

## Licence

[MIT](LICENSE).
