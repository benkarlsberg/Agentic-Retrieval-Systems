# Metrics by question category

- Run: `/workspace/Agentic-Retrieval-Systems/results/runs/live_hotpot150_equal_retrieval_20260915_071431`
- Dataset types from: `/workspace/Agentic-Retrieval-Systems/fixtures/datasets/hotpot_dev_slice_150.jsonl`

| Arch | Type | N | EM | F1 | soft_EM | contains_gold | soft_F1 | Abstention | Tokens | Calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_agent | multi_hop | 100 | 0.58 | 0.1114 | 0.0 | 0.58 | 0.3103 | 0.94 | 2754.87 | 5.0 |
| multi_agent | multi_passage | 30 | 0.9 | 0.0948 | 0.0 | 0.9 | 0.4626 | 1.0 | 2736.3 | 5.0 |
| multi_agent | single_hop | 15 | 0.6667 | 0.1361 | 0.0 | 0.6667 | 0.4003 | 1.0 | 2627.4 | 5.0 |
| multi_agent | unanswerable | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.75 | 2632.75 | 5.0 |
| rag | multi_hop | 100 | 0.45 | 0.1035 | 0.0 | 0.45 | 0.2509 | 0.82 | 340.76 | 1.0 |
| rag | multi_passage | 30 | 0.7333 | 0.0837 | 0.0 | 0.7333 | 0.3812 | 0.8333 | 336.33 | 1.0 |
| rag | single_hop | 15 | 0.6667 | 0.1285 | 0.0 | 0.6667 | 0.3812 | 1.0 | 338.4 | 1.0 |
| rag | unanswerable | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 282.75 | 1.0 |
| single_agent | multi_hop | 100 | 0.49 | 0.1063 | 0.0 | 0.48 | 0.2639 | 0.79 | 960.51 | 3.12 |
| single_agent | multi_passage | 30 | 0.7 | 0.0791 | 0.0 | 0.7 | 0.3664 | 0.8333 | 859.63 | 3.0 |
| single_agent | single_hop | 15 | 0.6 | 0.1295 | 0.0 | 0.6 | 0.3336 | 0.7333 | 600.2 | 2.53 |
| single_agent | unanswerable | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 76.25 | 1.12 |
