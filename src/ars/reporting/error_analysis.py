from __future__ import annotations

import json
from pathlib import Path

from ars.schema import Example, ExampleResult


def write_error_analysis(
    results: list[ExampleResult],
    examples: list[Example],
    path: Path,
) -> Path:
    by_id = {e.example_id: e for e in examples}
    rows = []
    for r in results:
        ex = by_id[r.example_id]
        em = r.metrics.get("em", 0.0)
        f1 = r.metrics.get("f1", 0.0)
        abst_ok = r.metrics.get("abstention_correct", 0.0)
        category = "ok"
        if not ex.is_answerable:
            category = "abstention_ok" if abst_ok >= 1.0 else "false_answer_on_unanswerable"
        elif r.abstained:
            category = "false_abstention"
        elif em < 1.0 and f1 < 0.5:
            category = "wrong_answer"
        elif em < 1.0:
            category = "partial_match"
        rows.append(
            {
                "example_id": r.example_id,
                "example_type": ex.example_type.value,
                "architecture": r.architecture.value,
                "category": category,
                "question": r.question,
                "prediction": r.answer,
                "gold": r.gold_answer,
                "em": em,
                "f1": f1,
                "stop_reason": r.stop_reason.value,
                "retrieval_calls": r.retrieval_calls,
                "model_calls": r.model_calls,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path
