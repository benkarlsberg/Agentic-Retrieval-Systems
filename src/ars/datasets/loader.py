"""Dataset loading — fixtures by default, adapters when real data available."""

from __future__ import annotations

import json
from pathlib import Path

from ars.schema import Example, ExampleType

REPO_ROOT = Path(__file__).resolve().parents[3]


def dataset_path(name: str = "mini_eval") -> Path:
    return REPO_ROOT / "fixtures" / "datasets" / f"{name}.jsonl"


def load_dataset(path: Path | str | None = None) -> list[Example]:
    p = Path(path) if path else dataset_path()
    examples: list[Example] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(
                Example(
                    example_id=obj["example_id"],
                    question=obj["question"],
                    gold_answer=obj.get("gold_answer"),
                    gold_doc_ids=obj.get("gold_doc_ids", []),
                    example_type=ExampleType(obj.get("example_type", "single_hop")),
                    aliases=obj.get("aliases", []),
                    metadata=obj.get("metadata", {}),
                    is_answerable=obj.get("is_answerable", True),
                )
            )
    return examples
