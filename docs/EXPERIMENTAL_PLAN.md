# Experimental Plan — Agentic Retrieval Systems (ARS)

## Research questions

1. **RQ1 (Quality under budget):** Under matched retrieval and/or token budgets, does bounded single-agent iterative retrieval improve answer quality (EM/F1) over conventional retrieve-once RAG on multi-hop and multi-passage questions?
2. **RQ2 (Coordination vs gain):** Does role-separated multi-agent retrieval (planner / specialized retrievers / critic / synthesizer) improve evidence coverage and citation fidelity enough to justify coordination overhead (extra model calls, latency, duplicate passages)?
3. **RQ3 (Failure modes):** How do architectures differ on unanswerable and ambiguous questions (abstention correctness vs false answers)?
4. **RQ4 (Retrieval modality):** How sensitive are architecture rankings to sparse (BM25), dense (fixture embeddings), and hybrid retrieval under the same corpus?

## Architectures under comparison

| ID | Name | Behavior |
|----|------|----------|
| A | Conventional RAG | One retrieve (`top_k`), one generate |
| B | Bounded single-agent | Loop: decide retrieve / reformulate / stop; hard budgets |
| C | Multi-agent | Planner → role retrievers → critic → synthesizer; structured messages |

## Fair-comparison rules

1. **Same corpus** for all architectures (checked-in `fixtures/corpus/mini_wiki.jsonl` or user-supplied).
2. **Same dataset split** and example IDs.
3. **Same model interface** (default: `DeterministicModel` for offline; optional OpenAI-compatible via env).
4. **Budget-matched modes:**
   - `equal_retrieval`: matched max retrieval calls / docs (RAG forced to 1 call).
   - `equal_token`: matched max token budget.
   - `unconstrained`: per-architecture defaults (still hard-capped).
5. **Identical metrics** and statistical procedure (paired bootstrap CIs on shared examples).
6. **No cherry-picking** of per-example regenerations; seed fixed in config.
7. **Traces required** (JSONL) for auditability of queries, docs, stops, costs.

## Budgets (defaults)

See `configs/budgets/default.yaml`:

- `max_steps`, `max_model_calls`, `max_docs`, `max_tokens`, `max_retrieval_calls`, `top_k`

Termination reasons are first-class (`StopReason` enum) and logged.

## Offline vs live evaluation policy

- **Default / CI / this repo’s checked-in results:** fixture corpus + `DeterministicModel`. Metrics are **offline-fixture**, not claims about frontier LLMs.
- **Optional live runs:** set `model.provider: openai` and `OPENAI_API_KEY`. Do not commit live secrets. Clearly label any live results separately from fixture baselines.

## Ablations (10)

Configs in `configs/ablations/`:

1. No critic  
2. No planner  
3. Single retriever role  
4. Hybrid retrieval  
5. Dense-only retrieval  
6. `top_k=1`  
7. `top_k=10`  
8. Strict budget  
9. No reformulate (single-agent)  
10. With reranker  

## Success metrics

- Answer: EM, token F1, lexical similarity (embedding optional later)
- Retrieval: recall@k, MRR
- Citations: precision/recall vs gold doc IDs
- Abstention correctness
- Efficiency: latency, model/retrieval calls, tokens, unique vs duplicate passages, estimated USD cost

## Statistical analysis

Paired bootstrap CIs (`ars.evaluation.stats.paired_bootstrap_ci`) on metric deltas between architecture pairs on the same example IDs.

## Deliverables

- Reproducible YAML configs + `scripts/run_experiment.py`
- Trace JSONL + HTML viewer
- Tables/plots under `results/`
- `docs/FINDINGS_DRAFT.md` (fixture-only until live runs exist)
