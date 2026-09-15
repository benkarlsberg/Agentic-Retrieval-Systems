# Results

> Distinguish **hypotheses** from **observed results**.  
> Numbers below are **offline-fixture** (`DeterministicModel` + mini corpus), not live LLM evals.

## Hypotheses

- H1: Under equal retrieval budget, single-agent ≥ RAG on multi-hop F1 (**live models**).
- H2: Multi-agent improves recall@k / citation recall vs RAG on multi-passage items, at higher model-call cost.
- H3: Multi-agent and single-agent improve abstention correctness vs naive RAG on unanswerable items.

## Observed results (offline-fixture, n=13)

Source: `results/runs/offline_fixture_compare_20260915_024627`

| Architecture | EM | F1 | Abstention | Model calls | Retrieval calls | Tokens |
|--------------|----|----|------------|-------------|-----------------|--------|
| rag | 0.846 | 0.810 | 1.000 | 1.00 | 1.00 | ~295 |
| single_agent | 0.846 | 0.810 | 1.000 | 3.00 | 1.00 | ~654 |
| multi_agent | 0.846 | 0.810 | 1.000 | 4.31 | 2.31 | ~1626 |

Interpretation: fixture run shows **matched heuristic quality** with
**clearly increasing coordination/token cost** RAG → single-agent → multi-agent.
See `docs/FINDINGS_DRAFT.md`.

## What this does *not* show

- Live LLM reasoning quality
- Production API latency
- Full HotpotQA/NQ ranking

## Next live experiments

1. `model.provider: openai` on a stratified sample with equal_retrieval / equal_token.
2. Run ablations in `configs/ablations/`.
3. Paired bootstrap on architecture deltas with larger n.
