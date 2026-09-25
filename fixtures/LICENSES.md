# Fixture licenses and provenance

## mini_wiki.jsonl / mini_eval.jsonl

- **Status**: Original synthetic/educational passages written for this repository.
- **Purpose**: Offline unit/integration tests and fixture experiments.
- **License**: MIT (same as this repository).
- **Not derived from**: HotpotQA, Natural Questions, or 2WikiMultihopQA copyrighted text
  beyond the separately attributed Hotpot slice below.
  Adapters exist to load those datasets *when the user provides local files*.

## HotpotQA slice: `datasets/hotpot_dev_slice_150.jsonl` and `corpus/hotpot_dev_slice_150.jsonl`

- **Source dataset**: HotpotQA distractor development set (`hotpot_dev_distractor_v1.json`).
- **Paper**: Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering"
  (EMNLP 2018). https://hotpotqa.github.io/
- **License**: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **Attribution**: This repository redistributes a **stratified ~150-example evaluation slice**
  and a BM25 passage corpus built only from those examples' provided contexts
  (seed=42; see `docs/VERSION_PINS.md` and `fixtures/raw/hotpot_dev_slice_150_build_meta.json`).
- **Download (public)**:
  - Mirror used for this build: `https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json`
  - Official: `http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json`
- **Synthetic unanswerable items** in the slice are original to this repo (MIT) and are not HotpotQA text.
- **Reuse**: Downstream use of Hotpot-derived text must comply with CC BY-SA 4.0 (attribution + share-alike).
- The full raw file (`fixtures/raw/hotpot_dev_distractor_v1.json`) is not redistributed; download it from the URLs above to rebuild the slice.

## External datasets (optional, user-supplied)

| Dataset | Typical license | Adapter |
|---------|-----------------|---------|
| HotpotQA | CC BY-SA 4.0 | `HotpotQAAdapter` |
| Natural Questions | CC BY-SA 3.0 | `NaturalQuestionsAdapter` |
| 2WikiMultihopQA | CC BY-SA 4.0 | `WikiMultihopAdapter` |

Apart from the HotpotQA slice above, users must download and cite those datasets themselves.

## Embeddings

Fixture dense vectors are produced by a deterministic hashing embedder (`FixtureEmbedder`).
No third-party embedding model weights are bundled.
