"""Adapters for HotpotQA / NQ / 2WikiMultihop-style datasets.

These load real JSON/JSONL when paths are provided; otherwise raise FileNotFoundError
so callers can fall back to fixtures. No network downloads in tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ars.schema import Example, ExampleType


class _BaseAdapter:
    name: str = "base"

    def load(self, path: Path, max_examples: int | None = None) -> list[Example]:
        if not path.exists():
            raise FileNotFoundError(
                f"{self.name} data not found at {path}. Use fixtures/datasets/mini_eval.jsonl offline."
            )
        raw = self._read(path)
        examples = [self._convert(i, row) for i, row in enumerate(raw)]
        if max_examples is not None:
            examples = examples[:max_examples]
        return examples

    def _read(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix == ".jsonl":
            rows = []
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            return rows
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return list(data)
        for key in ("data", "examples", "items"):
            if key in data and isinstance(data[key], list):
                return list(data[key])
        raise ValueError(f"Unrecognized {self.name} schema in {path}")

    def _convert(self, idx: int, row: dict[str, Any]) -> Example:
        raise NotImplementedError


class HotpotQAAdapter(_BaseAdapter):
    """HotpotQA-style: question, answer, supporting_facts / context."""

    name = "hotpotqa"

    def _convert(self, idx: int, row: dict[str, Any]) -> Example:
        gold_ids: list[str] = []
        sf = row.get("supporting_facts") or row.get("gold_doc_ids") or []
        if sf and isinstance(sf[0], (list, tuple)):
            gold_ids = [str(x[0]) for x in sf]
        elif sf:
            gold_ids = [str(x) for x in sf]
        qtype = row.get("type", "bridge")
        et = ExampleType.MULTI_HOP if qtype in ("bridge", "multi_hop") else ExampleType.MULTI_PASSAGE
        return Example(
            example_id=str(row.get("_id") or row.get("example_id") or f"hotpot_{idx}"),
            question=row["question"],
            gold_answer=row.get("answer"),
            gold_doc_ids=gold_ids,
            example_type=et,
            aliases=row.get("aliases", []),
            metadata={"source": "hotpotqa", "type": qtype},
            is_answerable=row.get("answer", "").lower() not in ("", "yes", "no")
            or bool(row.get("answer")),
        )


class NaturalQuestionsAdapter(_BaseAdapter):
    """NQ-style: question + short/long answers."""

    name = "natural_questions"

    def _convert(self, idx: int, row: dict[str, Any]) -> Example:
        answer = row.get("gold_answer") or row.get("short_answer") or row.get("answer")
        if isinstance(answer, list):
            answer = answer[0] if answer else None
        return Example(
            example_id=str(row.get("example_id") or row.get("id") or f"nq_{idx}"),
            question=row["question"],
            gold_answer=answer,
            gold_doc_ids=list(row.get("gold_doc_ids") or []),
            example_type=ExampleType.SINGLE_HOP,
            aliases=row.get("aliases", []),
            metadata={"source": "nq"},
            is_answerable=answer is not None and str(answer).strip() != "",
        )


class WikiMultihopAdapter(_BaseAdapter):
    """2WikiMultihopQA-style adapter."""

    name = "2wikimultihop"

    def _convert(self, idx: int, row: dict[str, Any]) -> Example:
        gold_ids = list(row.get("gold_doc_ids") or [])
        if not gold_ids and "supporting_facts" in row:
            sf = row["supporting_facts"]
            if sf and isinstance(sf[0], (list, tuple)):
                gold_ids = [str(x[0]) for x in sf]
        return Example(
            example_id=str(row.get("_id") or row.get("example_id") or f"wikihop_{idx}"),
            question=row["question"],
            gold_answer=row.get("answer") or row.get("gold_answer"),
            gold_doc_ids=gold_ids,
            example_type=ExampleType.MULTI_HOP,
            aliases=row.get("aliases", []),
            metadata={"source": "2wikimultihop"},
            is_answerable=True,
        )
