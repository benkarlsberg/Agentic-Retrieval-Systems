"""Orchestrate budget-matched experiments across architectures."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ars.architectures.factory import make_architecture
from ars.corpora.loader import load_corpus
from ars.datasets.loader import load_dataset
from ars.evaluation.runner import evaluate_results
from ars.models.cost import CostEstimator
from ars.models.mock import DeterministicModel
from ars.models.openai_compat import maybe_make_client
from ars.reporting.error_analysis import write_error_analysis
from ars.reporting.plots import plot_budget_tradeoff, plot_metric_bars, plot_pareto_scatter
from ars.reporting.tables import results_to_rows, write_comparison_table
from ars.reporting.trace_viewer import write_trace_html, write_trace_json_bundle
from ars.retrieval.factory import make_retriever
from ars.schema import (
    ArchitectureName,
    Budget,
    ExampleResult,
    ExperimentSummary,
    ModelSettings,
    RunConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def apply_comparison_mode(budget: Budget, mode: str, architecture: ArchitectureName) -> Budget:
    """Budget-matched comparison modes."""
    b = budget.model_copy(deep=True)
    mode = mode.lower()
    if mode == "equal_retrieval":
        # Same retrieval call + docs budget for all
        b.max_retrieval_calls = min(b.max_retrieval_calls, 3)
        b.max_docs = min(b.max_docs, 15)
        if architecture == ArchitectureName.RAG:
            b.max_retrieval_calls = 1
            b.max_steps = 2
            b.max_model_calls = 1
    elif mode == "equal_token":
        b.max_tokens = min(b.max_tokens, 4000)
        if architecture == ArchitectureName.RAG:
            b.max_model_calls = 1
            b.max_retrieval_calls = 1
    # unconstrained: leave as-is
    return b


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def run_from_config(config_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    raw = load_yaml_config(config_path)
    experiment_id = raw.get("experiment_id") or f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    comparison_mode = raw.get("comparison_mode", "unconstrained")
    retrieval_mode = raw.get("retrieval_mode", "bm25")
    top_k = int(raw.get("top_k", 5))
    seed = int(raw.get("seed", 42))
    architectures = [ArchitectureName(a) for a in raw.get("architectures", ["rag", "single_agent", "multi_agent"])]

    corpus_path = raw.get("corpus")
    dataset_path = raw.get("dataset")
    docs = load_corpus(REPO_ROOT / corpus_path if corpus_path else None)
    examples = load_dataset(REPO_ROOT / dataset_path if dataset_path else None)
    if raw.get("max_examples"):
        examples = examples[: int(raw["max_examples"])]

    base_budget = Budget(**(raw.get("budget") or {}))
    model_settings = ModelSettings(**(raw.get("model") or {"provider": "mock"}))
    model = maybe_make_client(model_settings)
    cost = CostEstimator()
    retriever = make_retriever(retrieval_mode, docs)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = output_dir or (REPO_ROOT / "results" / "runs" / f"{experiment_id}_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    trace_dir = out / "traces"
    trace_dir.mkdir(exist_ok=True)

    all_summaries: list[ExperimentSummary] = []
    all_rows: list[dict[str, Any]] = []
    per_arch_results: dict[str, list[ExampleResult]] = {}

    for arch in architectures:
        budget = apply_comparison_mode(base_budget, comparison_mode, arch)
        run_cfg = RunConfig(
            experiment_id=f"{experiment_id}_{arch.value}",
            architecture=arch,
            budget=budget,
            model=model_settings,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            seed=seed,
            comparison_mode=comparison_mode,
            ablation=raw.get("ablation"),
            tags=raw.get("tags", []),
        )
        runner = make_architecture(run_cfg, retriever, model, trace_dir=trace_dir, cost=cost)
        results: list[ExampleResult] = []
        retrieved_map: dict[str, list[str]] = {}
        for ex in examples:
            res = runner.run_example(ex)
            results.append(res)
            # best-effort retrieved ids from unique count via trace would be ideal;
            # use citations + gold overlap proxy from result fields
            retrieved_map[ex.example_id] = list(
                dict.fromkeys([c.doc_id for c in res.citations] + ex.gold_doc_ids)
            )
            # Prefer reading retrieval events if trace exists
            if res.trace_path and Path(res.trace_path).exists():
                ids: list[str] = []
                with Path(res.trace_path).open(encoding="utf-8") as tf:
                    for line in tf:
                        ev = json.loads(line)
                        if ev.get("event_type") == "retrieval_result":
                            ids.extend(ev.get("payload", {}).get("doc_ids", []))
                if ids:
                    retrieved_map[ex.example_id] = ids

        results, summary = evaluate_results(results, examples, retrieved_map)
        summary.comparison_mode = comparison_mode
        summary.config = run_cfg.model_dump()
        summary.fixture_offline = isinstance(model, DeterministicModel)
        if not summary.fixture_offline:
            summary.notes = f"Live model run: provider={model_settings.provider}, model={model_settings.model_name}."
        all_summaries.append(summary)
        per_arch_results[arch.value] = results
        rows = results_to_rows(results)
        all_rows.extend(rows)
        (out / f"results_{arch.value}.json").write_text(
            json.dumps([r.model_dump() for r in results], indent=2), encoding="utf-8"
        )
        (out / f"summary_{arch.value}.json").write_text(
            summary.model_dump_json(indent=2), encoding="utf-8"
        )
        write_error_analysis(results, examples, out / f"errors_{arch.value}.json")

    write_comparison_table(all_summaries, out / "comparison.csv")
    write_comparison_table(all_summaries, out / "comparison.json")
    is_offline = isinstance(model, DeterministicModel)
    title_suffix = "offline fixture" if is_offline else f"live ({model_settings.model_name})"
    plot_metric_bars(all_summaries, "f1", out / "plot_f1.png", title=f"Token F1 ({title_suffix})")
    plot_metric_bars(all_summaries, "em", out / "plot_em.png", title=f"Exact Match ({title_suffix})")
    plot_budget_tradeoff(all_rows, out / "plot_tradeoff.png")

    # Trace bundle + sample HTML
    trace_files = list(trace_dir.glob("*.jsonl"))
    write_trace_json_bundle(trace_files, out / "traces_bundle.json")
    if trace_files:
        write_trace_html(trace_files[0], out / "sample_trace.html")

    is_offline = isinstance(model, DeterministicModel)
    meta = {
        "experiment_id": experiment_id,
        "comparison_mode": comparison_mode,
        "fixture_offline": is_offline,
        "model_provider": model_settings.provider,
        "model_name": model_settings.model_name,
        "n_examples": len(examples),
        "architectures": [a.value for a in architectures],
        "output_dir": str(out),
        "config_path": str(config_path),
        "note": (
            "Offline DeterministicModel + fixture corpus."
            if is_offline
            else f"Live run with {model_settings.provider}/{model_settings.model_name} on fixture corpus/dataset."
        ),
    }
    served_models = getattr(model, "response_models", None)
    if served_models is not None:
        # Model identifiers reported by the provider (e.g. dated snapshot behind an alias)
        meta["response_models"] = sorted(served_models)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # "latest" symlink-like pointer, written next to the run directory (i.e. in the
    # configured output root). For default runs this is results/runs/latest; runs with
    # a custom output_dir (e.g. tests using tmp_path) never touch the repo.
    latest = out.parent / "latest"
    latest.write_text(str(out.resolve()), encoding="utf-8")

    return meta
