# Experiment Report

> **LABEL: offline-fixture results** — DeterministicModel / fixture corpus.
> Do not interpret as live LLM production performance.

- Experiment: `offline_fixture_compare`
- Comparison mode: `unconstrained`
- N examples: 13
- Architectures: rag, single_agent, multi_agent

## Aggregate metrics

| Architecture | EM | F1 | Abstention | Latency (ms) | Model calls | Retrieval calls |
|---|---:|---:|---:|---:|---:|---:|
| rag | 0.8461538461538461 | 0.8095238095238095 | 1.0 | 0.7810373846181252 | 1.0 | 1.0 |
| single_agent | 0.8461538461538461 | 0.8095238095238095 | 1.0 | 0.9648000000197499 | 3.0 | 1.0 |
| multi_agent | 0.8461538461538461 | 0.8095238095238095 | 1.0 | 1.3576280768680031 | 4.3076923076923075 | 2.3076923076923075 |

## Artifacts
- Run dir: `/workspace/Agentic-Retrieval-Systems/results/runs/offline_fixture_compare_20260915_024627`
