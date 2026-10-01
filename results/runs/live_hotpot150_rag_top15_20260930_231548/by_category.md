# Metrics by question category

- Run: `/workspace/Agentic-Retrieval-Systems/results/runs/live_hotpot150_rag_top15_20260930_231548`
- Dataset types from: `/workspace/Agentic-Retrieval-Systems/fixtures/datasets/hotpot_dev_slice_150.jsonl`

| Arch | Type | N | EM | F1 | soft_EM | contains_gold | soft_F1 | Abstention | Tokens | Calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rag | multi_hop | 100 | 0.61 | 0.1193 | 0.0 | 0.61 | 0.324 | 0.96 | 841.15 | 1.0 |
| rag | multi_passage | 30 | 0.8667 | 0.092 | 0.0 | 0.8667 | 0.4458 | 0.9667 | 806.6 | 1.0 |
| rag | single_hop | 15 | 0.7333 | 0.1031 | 0.0 | 0.7333 | 0.3997 | 1.0 | 811.87 | 1.0 |
| rag | unanswerable | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.875 | 736.62 | 1.0 |
