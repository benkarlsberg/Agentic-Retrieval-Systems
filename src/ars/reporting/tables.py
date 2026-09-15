from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ars.schema import ExampleResult, ExperimentSummary


def results_to_rows(results: list[ExampleResult]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        row = {
            "example_id": r.example_id,
            "architecture": r.architecture.value,
            "answer": r.answer,
            "gold_answer": r.gold_answer,
            "stop_reason": r.stop_reason.value,
            "abstained": r.abstained,
            "latency_ms": r.latency_ms,
            "model_calls": r.model_calls,
            "retrieval_calls": r.retrieval_calls,
            "unique_docs": r.unique_docs,
            "duplicate_docs": r.duplicate_docs,
            "tokens": r.prompt_tokens + r.completion_tokens,
            "cost_usd": r.estimated_cost_usd,
        }
        row.update({f"metric_{k}": v for k, v in r.metrics.items()})
        rows.append(row)
    return rows


def write_comparison_table(
    summaries: list[ExperimentSummary],
    path: Path,
    metrics: list[str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = metrics or ["em", "f1", "lexical_sim", "abstention_correct", "recall@k", "mrr", "latency_ms", "model_calls", "retrieval_calls", "tokens"]
    rows: list[dict[str, Any]] = []
    for s in summaries:
        row: dict[str, Any] = {
            "experiment_id": s.experiment_id,
            "architecture": s.architecture.value if s.architecture else "",
            "n_examples": s.n_examples,
            "fixture_offline": s.fixture_offline,
        }
        for m in metrics:
            row[m] = s.aggregate_metrics.get(m)
            row[f"{m}_ci_low"] = s.aggregate_metrics.get(f"{m}_ci_low")
            row[f"{m}_ci_high"] = s.aggregate_metrics.get(f"{m}_ci_high")
        rows.append(row)

    if path.suffix == ".json":
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    else:
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return path
