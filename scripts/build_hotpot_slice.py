#!/usr/bin/env python3
"""Rebuild stratified HotpotQA ~150 eval slice + BM25 corpus (seed=42)."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
RAW = REPO_ROOT / "fixtures/raw/hotpot_dev_distractor_v1.json"
OUT_DS = REPO_ROOT / "fixtures/datasets/hotpot_dev_slice_150.jsonl"
OUT_CORPUS = REPO_ROOT / "fixtures/corpus/hotpot_dev_slice_150.jsonl"
META_OUT = REPO_ROOT / "fixtures/raw/hotpot_dev_slice_150_build_meta.json"

UNANSWERABLE = [
    ("ua_synth_001", "Who is the CEO of the underwater Mars colony Atlantis Prime?"),
    ("ua_synth_002", "What is the atomic weight of the fictional element Quinium?"),
    ("ua_synth_003", "When did the city of New Zealandia declare independence from Antarctica?"),
    ("ua_synth_004", "Which moon of Jupiter is named after the unicorn goddess Zelphara?"),
    ("ua_synth_005", "Who invented the perpetual motion engine patented in 2099?"),
    ("ua_synth_006", "What language is spoken exclusively in the floating city of Aeropolis?"),
    ("ua_synth_007", "How many Olympic gold medals did the fictional athlete Zorp Glaxon win?"),
    ("ua_synth_008", "What is the capital of the Republic of Middle Earthia?"),
]


def sanitize_title(t: str) -> str:
    return t.replace("\n", " ").strip()


def ctx_sent_count(row: dict) -> int:
    return sum(len(sents) for _, sents in (row.get("context") or []))


def sample_unique(rng: random.Random, pool: list, k: int, exclude_ids: set) -> list:
    cand = [r for r in pool if r["_id"] not in exclude_ids]
    rng.shuffle(cand)
    return cand[:k]


def main() -> None:
    if not RAW.exists():
        raise SystemExit(
            f"Missing {RAW}. Download from "
            "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json"
        )
    data = json.loads(RAW.read_text(encoding="utf-8"))
    h = hashlib.sha256()
    with RAW.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    raw_sha = h.hexdigest()

    bridge = [r for r in data if r.get("type") == "bridge"]
    comparison = [r for r in data if r.get("type") == "comparison"]
    easy = sorted([r for r in data if r.get("level") == "easy"], key=ctx_sent_count)
    rng = random.Random(SEED)

    selected: list[tuple[str, dict]] = []
    exclude: set[str] = set()
    bridge_hard = [r for r in bridge if r.get("level") in ("hard", "medium")]
    bridge_rest = [r for r in bridge if r.get("level") == "easy"]
    b1 = sample_unique(rng, bridge_hard, 100, exclude)
    if len(b1) < 100:
        b1 += sample_unique(rng, bridge_rest, 100 - len(b1), exclude | {r["_id"] for r in b1})
    for r in b1:
        exclude.add(r["_id"])
        selected.append(("multi_hop", r))
    for r in sample_unique(rng, comparison, 30, exclude):
        exclude.add(r["_id"])
        selected.append(("multi_passage", r))
    e_pool = [r for r in easy if r["_id"] not in exclude]
    top = e_pool[:80]
    rng.shuffle(top)
    for r in top[:15]:
        exclude.add(r["_id"])
        selected.append(("single_hop", r))

    corpus: dict[str, dict] = {}
    examples_out: list[dict] = []
    for etype, row in selected:
        ex_id = row["_id"]
        title_sents: dict[str, list[tuple[int, str]]] = {}
        for title, sents in row.get("context") or []:
            title = sanitize_title(title)
            title_sents[title] = [(i, s) for i, s in enumerate(sents)]
            for i, s in enumerate(sents):
                text = s if isinstance(s, str) else str(s)
                base_id = f"{title}::s{i}"
                if base_id in corpus and corpus[base_id]["text"] != text:
                    doc_id = f"{ex_id}::{title}::s{i}"
                else:
                    doc_id = base_id
                corpus[doc_id] = {
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "metadata": {"source": "hotpotqa", "example_id": ex_id, "sent_id": i},
                }
        support_titles = list(
            dict.fromkeys(
                sanitize_title(str(item[0]))
                for item in (row.get("supporting_facts") or [])
                if isinstance(item, (list, tuple)) and item
            )
        )
        gold_doc_ids: list[str] = []
        for title in support_titles:
            for i, _ in title_sents.get(title, []):
                base_id = f"{title}::s{i}"
                if base_id in corpus:
                    gold_doc_ids.append(base_id)
                elif f"{ex_id}::{title}::s{i}" in corpus:
                    gold_doc_ids.append(f"{ex_id}::{title}::s{i}")
        examples_out.append(
            {
                "example_id": ex_id,
                "question": row["question"],
                "gold_answer": row.get("answer"),
                "gold_doc_ids": gold_doc_ids,
                "example_type": etype,
                "aliases": [],
                "metadata": {
                    "source": "hotpotqa",
                    "hotpot_type": row.get("type"),
                    "level": row.get("level"),
                    "supporting_titles": support_titles,
                },
                "is_answerable": True,
            }
        )

    for uid, q in UNANSWERABLE:
        examples_out.append(
            {
                "example_id": uid,
                "question": q,
                "gold_answer": None,
                "gold_doc_ids": [],
                "example_type": "unanswerable",
                "aliases": [],
                "metadata": {"source": "synthetic", "note": "intentionally unanswerable vs Hotpot corpus"},
                "is_answerable": False,
            }
        )

    rng.shuffle(examples_out)
    OUT_DS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DS.open("w", encoding="utf-8") as f:
        for ex in examples_out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    docs_sorted = sorted(corpus.values(), key=lambda d: d["doc_id"])
    OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CORPUS.open("w", encoding="utf-8") as f:
        for d in docs_sorted:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    type_counts = Counter(ex["example_type"] for ex in examples_out)
    meta = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "source_file": str(RAW.relative_to(REPO_ROOT)),
        "source_sha256": raw_sha,
        "source_bytes": RAW.stat().st_size,
        "dataset_id": "hotpot_qa/distractor/validation (hotpot_dev_distractor_v1.json)",
        "download_url": "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json",
        "official_url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "n_examples": len(examples_out),
        "example_type_counts": dict(type_counts),
        "corpus_docs": len(docs_sorted),
    }
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
