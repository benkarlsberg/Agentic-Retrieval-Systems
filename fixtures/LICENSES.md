# Fixture licenses and provenance

## mini_wiki.jsonl / mini_eval.jsonl

- **Status**: Original synthetic/educational passages written for this repository.
- **Purpose**: Offline unit/integration tests and fixture experiments.
- **License**: MIT (same as this repository).
- **Not derived from**: HotpotQA, Natural Questions, or 2WikiMultihopQA copyrighted text.
  Adapters exist to load those datasets *when the user provides local files*; we do not redistribute them.

## External datasets (optional, user-supplied)

| Dataset | Typical license | Adapter |
|---------|-----------------|---------|
| HotpotQA | CC BY-SA 4.0 | `HotpotQAAdapter` |
| Natural Questions | CC BY-SA 3.0 | `NaturalQuestionsAdapter` |
| 2WikiMultihopQA | CC BY-SA 4.0 | `WikiMultihopAdapter` |

Users must download and cite those datasets themselves. This repo ships only the synthetic mini fixtures.

## Embeddings

Fixture dense vectors are produced by a deterministic hashing embedder (`FixtureEmbedder`).
No third-party embedding model weights are bundled.
