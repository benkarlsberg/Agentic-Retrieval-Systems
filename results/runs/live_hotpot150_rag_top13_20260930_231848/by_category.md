# Metrics by question category

- Run: `/workspace/Agentic-Retrieval-Systems/results/runs/live_hotpot150_rag_top13_20260930_231848`
- Dataset types from: `/workspace/Agentic-Retrieval-Systems/fixtures/datasets/hotpot_dev_slice_150.jsonl`

| Arch | Type | N | EM | F1 | soft_EM | contains_gold | soft_F1 | Abstention | Tokens | Calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rag | multi_hop | 100 | 0.61 | 0.1187 | 0.0 | 0.61 | 0.3267 | 0.95 | 745.0 | 1.0 |
| rag | multi_passage | 30 | 0.9 | 0.0893 | 0.0 | 0.9 | 0.4623 | 1.0 | 714.1 | 1.0 |
| rag | single_hop | 15 | 0.7333 | 0.1012 | 0.0 | 0.7333 | 0.3993 | 1.0 | 714.67 | 1.0 |
| rag | unanswerable | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.875 | 654.62 | 1.0 |
