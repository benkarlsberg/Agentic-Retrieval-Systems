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

### Passage-budget control (RAG top_k 15 and 13)

Sources: `results/runs/live_hotpot150_rag_top15_20260930_231548`,
`results/runs/live_hotpot150_rag_top13_20260930_231848` (configs `live_hotpot150_rag_top{15,13}.yaml`).
The multi-agent pipeline read 13.1 unique passages per question (unconstrained: median 14, range 10-15, 15 on
47 of 153 questions; equal retrieval: mean 13.1, median 13) against RAG's 5.0. These runs repeat RAG on the same
153 questions with the same corpus, retriever, prompt, and model, changing only `top_k`. RAG does not consult
retrieval-call or document caps, so its unconstrained and equal-retrieval runs are procedurally identical (the two
top-5 runs retrieve identical passages for all 153 questions) and one run per `top_k` serves both modes. Served
model: `gpt-4o-mini-2024-07-18` (recorded in `meta.json`); judge prompt `judge-v2`, 306 judgments, 0 failures.

| System | Lenient EM [95% CI] | Judge [95% CI] | Tokens | Latency mean / median (s) | Est. cost (USD, 153 q) | Evidence recall | All support |
|--------|---------------------|----------------|-------:|---------------------------|-----------------------:|----------------:|------------:|
| rag, top_k=5 (unconstrained) | 0.510 [0.431, 0.588] | 0.601 [0.523, 0.680] | 336 | 0.81 / 0.77 | 0.019 | 0.366 | 0.483 |
| rag, top_k=13 | 0.647 [0.569, 0.725] | 0.752 [0.680, 0.817] | 731 | 1.11 / 1.01 | 0.042 | 0.591 | 0.766 |
| rag, top_k=15 | 0.641 [0.562, 0.719] | 0.752 [0.680, 0.817] | 826 | 1.09 / 1.00 | 0.047 | 0.618 | 0.786 |
| multi_agent (unconstrained) | 0.621 [0.542, 0.699] | 0.765 [0.699, 0.830] | 2,744 | 43.16 / 41.72 | 0.157 | 0.641 | 0.766 |

(CIs in this table come from the control section of `results.json`, which uses its own bootstrap stream, so the
multi-agent intervals can differ in the third decimal from the tables above.)

Paired differences (judge / lenient EM; discordant counts in parentheses):

| Comparison | Judge | Lenient EM |
|------------|-------|------------|
| rag k15 − rag k5 (unconstrained) | +0.150 [+0.092, +0.216] (25/2) | +0.131 [+0.078, +0.190] (20/0) |
| multi − rag k15 (unconstrained) | +0.013 [−0.033, +0.059] (8/6) | −0.020 [−0.059, +0.020] (3/6) |
| multi − rag k15 (equal retrieval) | +0.013 [−0.033, +0.059] (8/6) | −0.020 [−0.059, +0.020] (3/6) |
| rag k13 − rag k5 (unconstrained) | +0.150 [+0.092, +0.216] (25/2) | +0.137 [+0.085, +0.196] (21/0) |
| multi − rag k13 (unconstrained) | +0.013 [−0.026, +0.052] (6/4) | −0.026 [−0.065, +0.007] (2/6) |
| rag k5 unconstrained − rag k5 equal retrieval (identical runs) | +0.000 [−0.020, +0.020] (1/1) | +0.007 [+0.000, +0.020] (1/0) |

By stratum (judge, unconstrained), rag k15 − rag k5 / multi − rag k15: multi-hop +0.190 (19/0) / −0.020 (3/5);
multi-passage +0.133 (6/2) / +0.100 [−0.033, +0.233] (4/1); short-context 0 / +0.067 (1/0); unanswerable 0 / 0.
Top-15 RAG recovers a share of 0.92 (paired bootstrap 95% CI 0.67–1.26) of the multi-agent judge gain over top-5
RAG. A longer context did not measurably hurt: 2 questions went from correct to incorrect (0 under lenient EM), and
judge accuracy on answerable questions whose supporting paragraphs were all retrieved is 0.843 (top-5, n=70) vs.
0.842 (top-15, n=114). The harness `recall@k` field is valid for RAG at every `top_k` (it equals the trace-based
evidence recall); only the single-agent value is wrong (1.0 instead of 0.383).

A smaller live check on the 13-question fixture set is in
`results/runs/live_gpt4o_mini_compare_20260915_035757` (all three architectures at 0.692 lenient EM).

## What this does *not* show

- Results for models other than `gpt-4o-mini`
- Production API latency
- Full HotpotQA/NQ ranking

## Next experiments

1. Passage-budget-matched baselines by default (done for RAG top_k 15/13; see above) and the equal-token mode.
2. Additional models and a larger HotpotQA sample.
3. Constrained short-answer output so strict EM and token F1 are informative.
4. Run ablations in `configs/ablations/`.

See [`papers/arxiv/`](../papers/arxiv/) for the full analysis.
