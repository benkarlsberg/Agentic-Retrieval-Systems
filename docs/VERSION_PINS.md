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
`datasets` (Hugging Face) optional for alternate HotpotQA loaders — slice build used a direct JSON download.

## Fixture embeddings

No model weights pinned — `FixtureEmbedder` is deterministic hashing (dim=64).

## HotpotQA scaled eval slice (seed=42)

| Field | Value |
|-------|-------|
| Dataset | HotpotQA distractor validation (`hotpot_dev_distractor_v1.json`) |
| Official URL | http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json |
| Mirror URL used | https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json |
| Downloaded / built (UTC) | 2026-09-15 (see `fixtures/raw/hotpot_dev_slice_150_build_meta.json`) |
| SHA256 (`hotpot_dev_distractor_v1.json`) | `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8` |
| Source bytes | 61065698 |
| Slice outputs | `fixtures/datasets/hotpot_dev_slice_150.jsonl`, `fixtures/corpus/hotpot_dev_slice_150.jsonl` |
| Slice size | 153 examples (100 multi_hop / 30 multi_passage / 15 single_hop / 8 unanswerable) |
| Corpus size | 5675 passages (`{title}::s{i}`) |
| License | CC BY-SA 4.0 (HotpotQA); synthetic UA items MIT |

Rebuild: `scripts/build_hotpot_slice.py` (requires raw JSON under `fixtures/raw/`).
