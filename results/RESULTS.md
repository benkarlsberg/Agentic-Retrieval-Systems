# Results

> Distinguish **hypotheses** from **observed results**.  
> The fixture section uses `DeterministicModel` + mini corpus. The HotpotQA section uses live `gpt-4o-mini`.

## Hypotheses

- H1: Under equal retrieval budget, single-agent ≥ RAG on multi-hop F1 (**live models**).
- H2: Multi-agent improves recall@k / citation recall vs RAG on multi-passage items, at higher model-call cost.
- H3: Multi-agent and single-agent improve abstention correctness vs naive RAG on unanswerable items.

## Observed results (offline-fixture, n=13)

Source: `results/runs/offline_fixture_compare_20260915_024627`

| Architecture | EM | F1 | Abstention | Model calls | Retrieval calls | Tokens |
|--------------|----|----|------------|-------------|-----------------|--------|
| rag | 0.846 | 0.810 | 1.000 | 1.00 | 1.00 | ~294 |
| single_agent | 0.846 | 0.810 | 1.000 | 3.00 | 1.00 | ~654 |
| multi_agent | 0.846 | 0.810 | 1.000 | 4.31 | 2.31 | ~1625 |

Interpretation: fixture run shows **matched heuristic quality** with
**clearly increasing coordination/token cost** RAG → single-agent → multi-agent.
See `docs/FINDINGS_DRAFT.md`.

## Observed results (live gpt-4o-mini, HotpotQA slice, n=153)

Sources: `results/runs/live_hotpot150_unconstrained_20260915_042424`,
`results/runs/live_hotpot150_equal_retrieval_20260915_071431`; statistics recomputed by
`papers/arxiv/analysis/compute_tables.py` into `papers/arxiv/analysis/results.json`
(percentile bootstrap, 10,000 resamples, seed 42).

Slice: 153 questions from the HotpotQA distractor dev set (100 bridge / multi_hop, 30 comparison /
multi_passage, 15 short-context / single_hop, 8 synthetic unanswerable), BM25 over 5,675 sentence-level passages.

**Lenient EM**: 1 if the normalized prediction equals the normalized gold answer, or either contains the other
(harness field `em`). Strict EM (`soft_em`, normalized equality only) is 0.000 for every row because the
model answers in full sentences.

Unconstrained budget:

| Architecture | Lenient EM [95% CI] | Soft F1 | Abstention | Model calls | Retrieval calls | Tokens |
|--------------|---------------------|---------|------------|-------------|-----------------|--------|
| rag | 0.510 [0.431, 0.588] | 0.279 | 0.843 | 1.00 | 1.00 | 336 |
| single_agent | 0.497 [0.418, 0.575] | 0.270 | 0.797 | 2.99 | 1.16 | 892 |
| multi_agent | 0.621 [0.542, 0.693] | 0.333 | 0.935 | 5.00 | 3.00 | 2,744 |

Equal-retrieval budget:

| Architecture | Lenient EM [95% CI] | Soft F1 | Abstention | Model calls | Retrieval calls | Tokens |
|--------------|---------------------|---------|------------|-------------|-----------------|--------|
| rag | 0.503 [0.425, 0.582] | 0.276 | 0.850 | 1.00 | 1.00 | 337 |
| single_agent | 0.516 [0.438, 0.595] | 0.277 | 0.804 | 2.93 | 1.12 | 859 |
| multi_agent | 0.621 [0.542, 0.693] | 0.333 | 0.948 | 5.00 | 3.00 | 2,732 |

Paired differences in lenient EM (unconstrained): multi-agent − RAG +0.111 [+0.059, +0.163];
single-agent − RAG −0.013 [−0.065, +0.039]. Multi-agent uses 8.2x the tokens of RAG; single-agent uses 2.7x.
Per-category breakdowns are in each run's `by_category.md` and in the paper.

**LLM judge.** `scripts/judge_answers.py` regrades the saved answers (no re-runs) with `gpt-4o-mini` at
temperature 0 (returned model `gpt-4o-mini-2024-07-18`, prompt `judge-v2`). The judge does not see the
architecture. An answer is correct only if it commits to the gold entity or value (aliases and paraphrases
accepted; hedging, a different entity, an incidental mention, or an abstention are incorrect); on the 8
unanswerable questions it is correct if and only if the system abstains. Verdicts are cached as `judge_<arch>.json` in each
run directory: 918 judgments, 0 failures, 313,858 prompt + 17,181 completion tokens (~USD 0.06).

| Architecture | Judge, unconstrained [95% CI] | Judge, equal retrieval [95% CI] | Lenient 1 → judge 0 | Lenient 0 → judge 1 (answerable + unanswerable) |
|--------------|-------------------------------|---------------------------------|---------------------|------------------------------------------------|
| rag | 0.601 [0.523, 0.680] | 0.601 [0.523, 0.673] | 8 | 14 + 8 |
| single_agent | 0.575 [0.497, 0.654] | 0.582 [0.503, 0.660] | 12 | 16 + 8 |
| multi_agent | 0.765 [0.693, 0.830] | 0.765 [0.699, 0.830] | 4 | 18 + 8 |

(Flip counts are for the unconstrained run.) Paired differences in judge accuracy (unconstrained): multi-agent − RAG
+0.163 [+0.098, +0.235] (28 vs. 3 discordant); single-agent − RAG −0.026 [−0.092, +0.039] (11 vs. 15).
Equal retrieval: +0.163 [+0.105, +0.229] (27 vs. 2) and −0.020 [−0.085, +0.046] (11 vs. 14).
Audit of 60 sampled unconstrained judgments (30 judge/lenient disagreements, 30 agreements;
`judge_audit.csv`): judge agrees on 54/60 (Cohen's kappa 0.78), lenient EM on 36/60 (kappa 0.23).

A smaller live check on the 13-question fixture set is in
`results/runs/live_gpt4o_mini_compare_20260915_035757` (all three architectures at 0.692 lenient EM).

## What this does *not* show

- Results for models other than `gpt-4o-mini`
- Production API latency
- Full HotpotQA/NQ ranking

## Next experiments

1. Additional models and a larger HotpotQA sample.
2. Constrained short-answer output so strict EM and token F1 are informative.
3. Run ablations in `configs/ablations/`.

See [`papers/arxiv/`](../papers/arxiv/) for the full analysis.
