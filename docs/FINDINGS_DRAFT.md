# Findings Draft — Offline Fixture Runs Only

> **Note:** offline-fixture / DeterministicModel results.  
> These numbers come from the checked-in mini corpus (`fixtures/corpus/mini_wiki.jsonl`),
> mini eval set (`fixtures/datasets/mini_eval.jsonl`, n=13), and the heuristic
> `DeterministicModel`. They are **not** live LLM benchmark claims and must not be
> presented as production model performance.

**Primary run:** `results/runs/offline_fixture_compare_20260915_024627`  
**Mode:** `unconstrained` · retrieval: BM25 · seed: 42

## What the fixture runs show

| Architecture   | EM    | F1    | Abstention correct | Avg model calls | Avg retrieval calls | Avg tokens |
|----------------|------:|------:|-------------------:|----------------:|--------------------:|-----------:|
| rag            | 0.846 | 0.810 | 1.000              | 1.00            | 1.00                | ~295       |
| single_agent   | 0.846 | 0.810 | 1.000              | 3.00            | 1.00                | ~654       |
| multi_agent    | 0.846 | 0.810 | 1.000              | 4.31            | 2.31                | ~1626      |

Bootstrap 95% CIs are in `comparison.json` (wide, as expected for n=13).

### Process / efficiency (the signal this offline run is good for)

1. **Answer quality was matched across architectures** on this tiny fixture once the mock
   answerer was question-gated. That is expected: the mock extracts facts from retrieved
   context with the same heuristics, so architecture differences show up mainly in
   **control flow and cost**, not in frontier reasoning.
2. **Cost / coordination overhead is ordered as designed:**
   RAG ≪ single-agent ≪ multi-agent on model calls and tokens.
   Multi-agent also issues more retrieval calls (~2.3 vs 1.0) and logs
   `coordination_messages` in traces.
3. **Abstention heuristics worked** on both unanswerable items for all three architectures
   (`abstention_correct = 1.0`). This validates the evaluation wiring for RQ3-style
   analysis; it does **not** prove live LLMs will abstain correctly.
4. **Budgets and traces are enforceable and auditable:** every example wrote JSONL events
   (queries, doc IDs+scores, routing, termination, answer). Sample HTML viewer:
   `sample_trace.html` in the run directory.
5. **Equal-retrieval mode** also completed successfully
   (`results/runs/equal_retrieval_budget_20260915_024643`), exercising budget-matched
   comparison without requiring API keys.

### Error analysis snapshot (offline)

- 11/13 answerable-or-partial items scored as `ok` under EM/F1 thresholds after mock tuning.
- 2/2 unanswerable items: `abstention_ok` for all architectures.
- Remaining misses (if any on other configs) tend to be multi-hop / multi-passage items
  where one-shot BM25 context lacks the second hop — exactly the regime iterative and
  multi-agent designs are meant to stress under **live** models.

## What this framework is designed to measure (next, with live models)

These are **research insights the harness enables**, not claims from the fixture numbers:

1. **RQ1:** Under `equal_retrieval` / `equal_token`, does single-agent iterative reformulation
   improve multi-hop F1 vs RAG?
2. **RQ2:** Does multi-agent role separation improve gold-evidence recall / citation fidelity
   enough to offset extra model calls and duplicate passages?
3. **RQ3:** Architecture gaps on unanswerable vs ambiguous questions (false answers vs false abstentions).
4. **RQ4:** Stability of architecture ranking under BM25 vs dense vs hybrid
   (`configs/ablations/04_hybrid_retrieval.yaml`, `05_dense_only.yaml`).

Statistical tool already wired: paired bootstrap CIs on per-example metric deltas.

## Explicit non-claims

- No frontier LLM quality ranking.
- No HotpotQA/NQ leaderboard numbers (adapters only; data not shipped).
- Ablation YAMLs are present for the 10 planned ablations; some behavioral toggles
  (e.g., hard `skip_critic`) are reserved extension points documented in config notes.

## Artifacts

- Tables: `results/tables/offline_fixture_compare.{csv,json}`
- Plots: `results/plots/offline_fixture_{em,f1,tradeoff}.png`
- Full run: `results/runs/offline_fixture_compare_20260915_024627/`
