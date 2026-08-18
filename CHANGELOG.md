# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/).

## [0.5.0]

### Added
- Cross-encoder reranking as an optional second stage: candidates from the
  fast first pass are rescored by a model that reads query and passage
  together, the architecture production RAG systems use.
- Content-addressed index cache, so a rebuilt corpus is never re-embedded.
  The key digests document bytes and index settings, which makes serving a
  stale index impossible. Nothing is stored with `pickle`.
- `rag-eval --compare` evaluates every configuration and ranks them.
- `state_dict` / `load_state_dict` on every retriever, so persistence is part
  of the retriever contract rather than a special case.
- Chunking now cuts on whitespace, so passages neither end nor begin in the
  middle of a word.

### Changed
- The project now speaks English throughout — code, tests, documentation and
  automation — matching how the wider ecosystem is used.
- The golden set grew from 6 to 12 questions over a three-document corpus.

### Fixed
- An evaluation key depended on a line break in the source document, so
  reformatting the corpus would silently have broken the measurement.
- The index cache returned a corrupt state when its array archive was missing,
  instead of reporting a miss.
- Temporary archives were written under a name NumPy then renamed, leaving the
  cache file unwritten.
- `rag-eval --compare` crashed when the embeddings extra was absent instead of
  reporting the configurations it could not measure.

## [0.4.0]

### Added
- Hybrid search through Reciprocal Rank Fusion.
- CodeQL analysis, Dependabot updates and a scheduled quality evaluation.
- Contribution guide, security policy and issue templates.

## [0.3.0]

### Added
- Dense retrieval with neural embeddings and lazy model loading.
- `Retriever` contract so strategies are interchangeable.
- Evaluation with Hit@k, MRR and Recall@k, plus a `rag-eval` command whose
  thresholds fail CI on regression.
- Type checking with mypy.

### Fixed
- The documented `.env` file was never loaded, so LLM mode never activated.
- Unreadable documents propagated a bare exception with no context.

## [0.2.0]

### Added
- HTTP API with FastAPI and a Docker deployment.
- Test matrix across Python 3.10, 3.11 and 3.12.

## [0.1.0]

### Added
- End-to-end RAG pipeline, runnable locally without keys.
