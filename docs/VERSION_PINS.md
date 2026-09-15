# Version pins and dependency notes

## Python

- Requires Python `>=3.10` (developed/tested on 3.13 in the build environment).

## Core runtime (see `pyproject.toml`)

| Package | Constraint | Role |
|---------|------------|------|
| pydantic | >=2.5 | Typed schemas |
| pyyaml | >=6.0 | Experiment configs |
| numpy | >=1.24 | Dense retrieval / bootstrap |
| rank-bm25 | >=0.2.2 | BM25 (pure-Python fallback included) |
| matplotlib | >=3.7 | Plots |
| jinja2 | >=3.1 | Optional templating |
| httpx | >=0.25 | Optional HTTP |

## Dev

pytest, pytest-cov, ruff, mypy, types-PyYAML

## Optional

`openai` extra for OpenAI-compatible live clients (not required for tests).

## Fixture embeddings

No model weights pinned — `FixtureEmbedder` is deterministic hashing (dim=64).
