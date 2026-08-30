# Working on this repository

For whoever does the work next — a person, Claude, Cursor. It is not a style
guide. It is the set of things this repository has already got wrong, and the
reasons behind the safeguards that look arbitrary until you know the story.

## The floor

**Python 3.10 is the promise.** `pyproject.toml` declares
`requires-python = ">=3.10"` and the matrix tests 3.10, 3.11 and 3.12. Anything
that entered the standard library after 3.10 is a bug here even when it runs
perfectly on your machine.

**Nothing is green until CI says so.** Not "tests pass locally", not "lint is
clean". The run on GitHub is the evidence; everything before it is a guess that
happens to be usually right.

**Verify against the pinned versions, not the installed ones.** `ruff` and
`mypy` are pinned to exact versions in `[project.optional-dependencies].dev`
precisely so local and CI agree. A local ruff a few minors behind will pass
files that CI rejects — that happened here in August 2026, and the commit
message claiming verification named a version the project no longer pinned.

## The gates

All of these run in CI. Run them before pushing, in this order:

```bash
pip install -e ".[dev]"
ruff format --check .   # formatting
ruff check .            # E, F, I, UP, B
mypy                    # rag_agent only
pytest                  # unit suite; integration is opt-in
rag-eval --min-hit-rate 0.90 --min-mrr 0.80
docker build -t rag-agent-lab:ci .
```

A tag runs every one of them again and refuses to publish if any fails, so a
release cannot be greener than main.

## Traps this repository has already fallen into

**`import tomllib` broke the 3.10 build.** It only entered the standard library
in 3.11. The release workflow happens to run 3.12, so the failure was invisible
there and would have surfaced for the first time on someone else's machine.
Ruff was the tell: it sorted the import into the third-party block, which looks
like a formatting nit and is not. Reading one declared field did not need a
TOML parser. (`9476b83` broke it, `64e4373` fixed it.)

**`python_version` in the mypy config broke CI.** It was pinned to 3.10 while
CI runs mypy on 3.12; analysing with a version other than the interpreter's
left mypy unable to read numpy's stubs, which use newer syntax. The setting is
**deliberately absent** and there is a comment saying so. Compatibility with
3.10 is guaranteed by the test matrix and by ruff, not by that pin. Do not
helpfully add it back. (`d38b188`.)

**`pythonpath = ["."]` in the pytest config is load-bearing.** CI runs bare
`pytest`, not `python -m pytest`, so the repository root is not on `sys.path`
and `import rag_agent` fails without it. (`a13d6cf`.)

**An evaluation key depended on a line break.** A golden-set answer was matched
against text that only existed because of where the source document happened to
wrap. Reformatting the corpus would have broken the measurement silently — the
worst failure mode a benchmark has, because the number stays plausible.

**NumPy appends `.npz`.** A temporary archive was written under a path the code
then tried to move, and the file it moved never existed. Cache writes are the
place this bites, because the failure looks like a cache miss.

**Optional extras must degrade, not crash.** `rag-eval --compare` used to raise
`ImportError` when the `embeddings` extra was absent, contradicting its own
docstring. It now reports which rows it could not measure and prints the rest.
Heavy dependencies are imported lazily so anyone on the lexical backend never
downloads PyTorch; keep it that way.

## Rules that are not negotiable

**Never `pickle`.** Loading one executes arbitrary code, and the index cache is
exactly the file an attacker would replace. Passages are JSON, vectors are a
NumPy archive, both inert.

**Never lower a threshold to make a build pass.** `--min-hit-rate` and
`--min-mrr` exist to fail. If a change genuinely improves the system and moves a
number, say so in the commit message with both values. A silently relaxed
threshold turns the whole evaluation into decoration.

**Never edit `data/quality-history.csv` by hand.** It is appended by
`scripts/record_quality.py` from a real measurement. A retro-fitted row is a
fabricated experiment.

**Dependencies are added reluctantly, and heavy ones are optional extras.** The
point of this project is that it can be read end to end.

## Dependabot

Routine minor and patch updates arrive grouped and weekly; CI decides whether
they land. Majors of `ruff`, `mypy`, `pytest` and the Docker `python` base image
are ignored on purpose — each of those is a migration with its own
verification, not something to merge because a bot opened it overnight.

One thing Dependabot cannot see: a version written as a plain string inside a
workflow's `with:` block is not a dependency, so nothing watches it. Those are
checked by hand or they rot.

## Commit messages

Say what changed and why it was wrong before. The diff already shows what moved;
the message is the only place the reason survives. If a fix defends against
something subtle, name the failure it prevents — the next person to read that
line will otherwise assume it was arbitrary and remove it.
