# Metrics by question category

- Run: `/workspace/Agentic-Retrieval-Systems/results/runs/live_hotpot150_unconstrained_20260915_042424`
- Dataset types from: `/workspace/Agentic-Retrieval-Systems/fixtures/datasets/hotpot_dev_slice_150.jsonl`

| Arch | Type | N | EM | F1 | soft_EM | contains_gold | soft_F1 | Abstention | Tokens | Calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_agent | multi_hop | 100 | 0.58 | 0.1168 | 0.0 | 0.58 | 0.3111 | 0.92 | 2763.49 | 5.0 |
| multi_agent | multi_passage | 30 | 0.8667 | 0.0901 | 0.0 | 0.8667 | 0.4495 | 1.0 | 2747.87 | 5.0 |
| multi_agent | single_hop | 15 | 0.7333 | 0.1368 | 0.0 | 0.7333 | 0.4213 | 1.0 | 2668.13 | 5.0 |
| multi_agent | unanswerable | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.75 | 2619.12 | 5.0 |
| rag | multi_hop | 100 | 0.46 | 0.1064 | 0.0 | 0.46 | 0.2551 | 0.81 | 339.76 | 1.0 |
| rag | multi_passage | 30 | 0.7333 | 0.085 | 0.0 | 0.7333 | 0.3812 | 0.8333 | 336.23 | 1.0 |
| rag | single_hop | 15 | 0.6667 | 0.113 | 0.0 | 0.6667 | 0.3812 | 1.0 | 339.27 | 1.0 |
| rag | unanswerable | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 282.75 | 1.0 |
| single_agent | multi_hop | 100 | 0.47 | 0.108 | 0.0 | 0.47 | 0.2583 | 0.78 | 989.45 | 3.17 |
| single_agent | multi_passage | 30 | 0.7 | 0.0772 | 0.0 | 0.7 | 0.3661 | 0.8333 | 902.77 | 3.07 |
| single_agent | single_hop | 15 | 0.5333 | 0.1094 | 0.0 | 0.5333 | 0.2991 | 0.7333 | 651.93 | 2.6 |
| single_agent | unanswerable | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 76.25 | 1.12 |
