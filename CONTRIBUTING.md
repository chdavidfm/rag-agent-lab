# Contributing

Thanks for the interest. This project favours readable code and demonstrable
quality: anything that lands should be runnable and measurable.

## Set up

```bash
git clone https://github.com/chdavidfm/rag-agent-lab.git
cd rag-agent-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

These four checks are exactly what CI runs. If they pass locally, they pass on
the server:

```bash
ruff format .   # apply formatting
ruff check .    # lint
mypy            # types
pytest          # test suite
```

Tests marked `integration` download real models and stay out of the default
run:

```bash
pytest -m integration
```

## Retrieval quality

Any change to how passages are found must be measured. CI rejects changes that
degrade the metrics:

```bash
rag-eval --min-hit-rate 0.90 --min-mrr 0.80
rag-eval --compare              # every configuration, side by side
```

If your change **improves** the numbers, raise the thresholds in
`.github/workflows/ci.yml` so the gain is locked in.

## Code standards

- Each module owns one responsibility and states it in its docstring.
- Heavy dependencies are imported lazily and live behind an optional extra,
  never in the core.
- Every public function carries type annotations.
- Comments explain **why**, not **what** — the code already says what.
- Tests use injected doubles rather than network calls, so the suite stays
  fast and deterministic.

## Commit messages

An imperative first line under 72 characters, and a body explaining the reason
whenever it is not obvious from the diff.

```
Add hybrid retrieval fusion

Lexical and dense search fail on different queries. RRF merges them by rank,
so no score calibration is needed.
```
