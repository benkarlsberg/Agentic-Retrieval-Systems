# Agentic Retrieval Systems (ARS)

Reproducible research framework comparing **conventional RAG**, **bounded single-agent iterative retrieval**, and **role-separated multi-agent retrieval**.

> **Default evaluation is offline-fixture** (`DeterministicModel` + checked-in mini corpus).  
> Do not read fixture metrics as live LLM benchmark claims.

## Research questions (short)

1. Does iterative retrieval beat retrieve-once RAG under **matched budgets**?
2. Does multi-agent role separation buy evidence/citation gains worth the **coordination overhead**?
3. How do architectures handle **unanswerable / ambiguous** questions?
4. How sensitive are rankings to **BM25 / dense / hybrid** retrieval?

See [docs/EXPERIMENTAL_PLAN.md](docs/EXPERIMENTAL_PLAN.md).

## Architecture overview

```
┌─────────────┐   ┌──────────────────────┐   ┌─────────────────────────────┐
│  RAG        │   │  Single-agent        │   │  Multi-agent                │
│  retrieve→  │   │  decide loop:        │   │  planner → retrievers →     │
│  generate   │   │  retrieve/reformulate│   │  critic → synthesizer       │
│             │   │  /stop + budgets     │   │  + coordination traces      │
└─────────────┘   └──────────────────────┘   └─────────────────────────────┘
         shared corpus · shared metrics · JSONL traces · budget modes
```

## Quickstart

```bash
cd Agentic-Retrieval-Systems
pip install -e ".[dev]"
make test
python scripts/run_experiment.py --config configs/experiments/offline_fixture_compare.yaml
python scripts/generate_report.py --run-dir results/runs/latest
```

## Package layout

```
src/ars/           # library (agents, architectures, retrieval, evaluation, tracing, …)
configs/           # experiments, budgets, ablations
fixtures/          # mini corpus + dataset (offline)
scripts/           # CLI runners
tests/             # unit + integration + fixture tests
docs/              # experimental plan, findings draft, pins
results/           # run outputs (generated)
```

## Reproduction commands

```bash
# Full offline comparison (3 architectures)
python scripts/run_experiment.py --config configs/experiments/offline_fixture_compare.yaml

# Budget-matched modes
python scripts/run_experiment.py --config configs/experiments/equal_retrieval.yaml
python scripts/run_experiment.py --config configs/experiments/equal_token.yaml

# Example ablation
python scripts/run_experiment.py --config configs/ablations/04_hybrid_retrieval.yaml

# Lint / typecheck / tests
make ci
```

Optional live OpenAI-compatible runs (not required):

```bash
export OPENAI_API_KEY=...
# edit config model.provider: openai
```

## Metrics

EM, token F1, lexical similarity, recall@k, MRR, citation P/R, abstention correctness, latency, call counts, tokens, unique vs duplicate passages, config-based cost estimates.

## Limitations

- Mock model answers are heuristic — useful for wiring/eval pipelines, **not** for claiming SOTA.
- Mini fixtures are synthetic (MIT); HotpotQA/NQ/2Wiki adapters load **user-provided** local files only.
- Some ablation YAML flags document intent; a few behavioral toggles are reserved for extension (see findings).
- Dense retriever uses hashing embeddings, not trained dual encoders.

## Citation / license

MIT. Fixture provenance: [fixtures/LICENSES.md](fixtures/LICENSES.md).


## Research paper (arXiv)

LaTeX source, figures, and PDF: [`papers/arxiv/`](papers/arxiv/).  
Upload guide: [`papers/arxiv/README.md`](papers/arxiv/README.md).  
Prebuilt PDF: [`papers/arxiv/main.pdf`](papers/arxiv/main.pdf).  
Source zip for arXiv: [`papers/arxiv/arxiv-source.zip`](papers/arxiv/arxiv-source.zip).
