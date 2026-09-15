from __future__ import annotations

from collections import defaultdict

from ars.evaluation.metrics import compute_example_metrics
from ars.evaluation.stats import mean_ci
from ars.schema import Example, ExampleResult, ExperimentSummary


def evaluate_results(
    results: list[ExampleResult],
    examples: list[Example],
    retrieved_map: dict[str, list[str]] | None = None,
) -> tuple[list[ExampleResult], ExperimentSummary]:
    by_id = {e.example_id: e for e in examples}
    retrieved_map = retrieved_map or {}
    for r in results:
        ex = by_id[r.example_id]
        r.metrics = compute_example_metrics(r, ex, retrieved_map.get(r.example_id))

    agg: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for k, v in r.metrics.items():
            agg[k].append(float(v))

    aggregate: dict[str, float] = {}
    for k, vals in agg.items():
        ci = mean_ci(vals)
        aggregate[k] = ci.mean
        aggregate[f"{k}_ci_low"] = ci.low
        aggregate[f"{k}_ci_high"] = ci.high

    arch = results[0].architecture if results else None
    summary = ExperimentSummary(
        experiment_id=results[0].experiment_id if results else "empty",
        architecture=arch,  # type: ignore[arg-type]
        n_examples=len(results),
        comparison_mode="see_config",
        aggregate_metrics=aggregate,
        fixture_offline=True,
        notes="Metrics from offline fixture/mock runs — not live LLM eval.",
    )
    return results, summary
