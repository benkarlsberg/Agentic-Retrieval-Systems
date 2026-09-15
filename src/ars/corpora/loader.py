"""Corpus loading utilities."""

from __future__ import annotations

import json
from pathlib import Path

from ars.schema import Document

REPO_ROOT = Path(__file__).resolve().parents[3]


def corpus_path(name: str = "mini_wiki") -> Path:
    return REPO_ROOT / "fixtures" / "corpus" / f"{name}.jsonl"


def load_corpus(path: Path | str | None = None) -> list[Document]:
    p = Path(path) if path else corpus_path()
    docs: list[Document] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(
                Document(
                    doc_id=obj["doc_id"],
                    title=obj.get("title", ""),
                    text=obj["text"],
                    metadata=obj.get("metadata", {}),
                )
            )
    return docs
