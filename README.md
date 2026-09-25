# Agentic Retrieval Systems (ARS)

Reproducible research framework comparing **conventional RAG**, **bounded single-agent iterative retrieval**, and **role-separated multi-agent retrieval**.

> The offline fixture results use a deterministic model on a small synthetic corpus and only validate the harness.
> Live `gpt-4o-mini` results on a 153-question HotpotQA slice are summarized [below](#hotpotqa-slice-results-live).

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

`results/runs/latest` is a local, untracked pointer file written by each run (next to the
run directory) containing the path of the most recent run; `generate_report.py` follows it.

## Package layout

```
src/ars/           # library (agents, architectures, retrieval, evaluation, tracing, …)
configs/           # experiments, budgets, ablations
fixtures/          # mini corpus + dataset (offline), HotpotQA dev slice
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

Optional live OpenAI-compatible runs (not required for tests). With `model.provider: openai` the client
fails if `OPENAI_API_KEY` is unset instead of falling back to the mock model:

```bash
export OPENAI_API_KEY=...
python scripts/run_experiment.py --config configs/experiments/live_hotpot150_unconstrained.yaml
python scripts/run_experiment.py --config configs/experiments/live_hotpot150_equal_retrieval.yaml
python scripts/report_by_category.py --run-dir results/runs/<run_dir>
```

The HotpotQA slice is checked in; `scripts/build_hotpot_slice.py` rebuilds it from the raw dev file
(see [docs/VERSION_PINS.md](docs/VERSION_PINS.md)).

## Metrics

EM (lenient), strict EM (`soft_em`), contains-gold, token F1, soft F1, lexical similarity, recall@k, MRR, citation P/R, abstention correctness, latency, call counts, tokens, unique vs duplicate passages, config-based cost estimates.

## HotpotQA-slice results (live)

`gpt-4o-mini`, BM25 over 5,675 sentence-level passages, 153 questions from the HotpotQA distractor dev set
(100 bridge, 30 comparison, 15 short-context, 8 synthetic unanswerable). Lenient EM is 1 when the normalized
prediction equals the normalized gold answer or either one contains the other. 95% bootstrap CIs in brackets.

| Architecture | Lenient EM (unconstrained) | Lenient EM (equal retrieval) | Tokens / question (unconstrained) | Model calls |
|---|---|---|---:|---:|
| RAG | 0.510 [0.431, 0.588] | 0.503 | 336 | 1.0 |
| Single-agent | 0.497 [0.418, 0.575] | 0.516 | 892 | 3.0 |
| Multi-agent | 0.621 [0.542, 0.693] | 0.621 | 2,744 | 5.0 |

Multi-agent beats RAG by +0.111 lenient EM (paired 95% CI +0.059 to +0.163) at 8.2x the tokens. The single-agent
loop uses 2.7x the tokens of RAG with no measurable gain. Strict EM is 0 for every architecture because answers
are verbose. This is one model on a small derived slice, not a HotpotQA leaderboard result.

Run artifacts: `results/runs/live_hotpot150_*`. Details: [results/RESULTS.md](results/RESULTS.md) and the paper in
[`papers/arxiv/`](papers/arxiv/).

## Limitations

- Mock model answers are heuristic — useful for wiring/eval pipelines, **not** for claiming SOTA.
- Mini fixtures are synthetic (MIT). The HotpotQA slice is CC BY-SA 4.0 (see [fixtures/LICENSES.md](fixtures/LICENSES.md)); NQ/2Wiki adapters load **user-provided** local files only.
- Live results cover one model (`gpt-4o-mini`) and 153 questions.
- Some ablation YAML flags document intent; a few behavioral toggles are reserved for extension (see findings).
- Dense retriever uses hashing embeddings, not trained dual encoders.

## Citation / license

MIT. Fixture provenance: [fixtures/LICENSES.md](fixtures/LICENSES.md).


## Paper

Draft paper (LaTeX + PDF): [`papers/arxiv/`](papers/arxiv/).
