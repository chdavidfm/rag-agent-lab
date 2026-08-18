## What changes

<!-- One or two sentences on the change and why it is needed. -->

## How to verify

<!-- Concrete steps a reviewer can follow. -->

## Checks

- [ ] `ruff format .` and `ruff check .` are clean
- [ ] `mypy` reports no errors
- [ ] `pytest` passes
- [ ] `rag-eval` does not regress the metrics (if retrieval is affected)
