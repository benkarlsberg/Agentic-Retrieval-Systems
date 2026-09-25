#!/usr/bin/env python3
"""Per-example_type metric tables + Pareto scatters from a run directory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ars.reporting.plots import plot_pareto_scatter  # noqa: E402


QUALITY_KEYS = ("em", "f1", "soft_em", "contains_gold", "soft_f1", "abstention_correct")


def _load_examples(dataset_path: Path | None) -> dict[str, str]:
    """Map example_id -> example_type."""
    mapping: dict[str, str] = {}
    if dataset_path is None or not dataset_path.exists():
        return mapping
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[row["example_id"]] = row.get("example_type", "unknown")
    return mapping


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def build_category_rows(
    results: list[dict[str, Any]],
    type_map: dict[str, str],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        et = type_map.get(r.get("example_id", ""), r.get("example_type") or "unknown")
        # metrics may be nested or flattened
        metrics = r.get("metrics") or {}
        flat = {**r, **{f"metric_{k}": v for k, v in metrics.items()}}
        for k, v in metrics.items():
            flat.setdefault(k, v)
        arch = r.get("architecture")
        if hasattr(arch, "value"):
            arch = arch.value
        buckets[(str(arch), et)].append(flat)

    out: list[dict[str, Any]] = []
    for (arch, et), rows in sorted(buckets.items()):
        row: dict[str, Any] = {
            "architecture": arch,
            "example_type": et,
            "n": len(rows),
        }
        for q in QUALITY_KEYS:
            vals = []
            for r in rows:
                if q in r:
                    vals.append(float(r[q]))
                elif f"metric_{q}" in r:
                    vals.append(float(r[f"metric_{q}"]))
                elif isinstance(r.get("metrics"), dict) and q in r["metrics"]:
                    vals.append(float(r["metrics"][q]))
            row[q] = round(_mean(vals), 4)
        for cost in ("tokens", "model_calls", "retrieval_calls", "latency_ms"):
            vals = []
            for r in rows:
                if cost == "tokens":
                    if "tokens" in r:
                        vals.append(float(r["tokens"]))
                    elif "prompt_tokens" in r:
                        vals.append(float(r.get("prompt_tokens", 0)) + float(r.get("completion_tokens", 0)))
                    elif f"metric_{cost}" in r:
                        vals.append(float(r[f"metric_{cost}"]))
                elif cost in r:
                    vals.append(float(r[cost]))
                elif f"metric_{cost}" in r:
                    vals.append(float(r[f"metric_{cost}"]))
            row[cost] = round(_mean(vals), 2)
        out.append(row)
    return out


def collect_per_example_rows(run_dir: Path, type_map: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("results_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for r in data:
            metrics = r.get("metrics") or {}
            arch = r.get("architecture")
            if isinstance(arch, dict):
                arch = arch.get("value", arch)
            eid = r.get("example_id", "")
            row = {
                "example_id": eid,
                "architecture": arch,
                "example_type": type_map.get(eid, "unknown"),
                "tokens": float(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)),
                "model_calls": float(r.get("model_calls", 0)),
                "retrieval_calls": float(r.get("retrieval_calls", 0)),
                "latency_ms": float(r.get("latency_ms", 0)),
            }
            for q in QUALITY_KEYS:
                row[q] = float(metrics.get(q, 0.0))
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional JSONL to recover example_type (defaults to meta/config or mini/hotpot slice)",
    )
    args = parser.parse_args()
    run_dir = args.run_dir
    if run_dir.name == "latest" and run_dir.is_file():
        run_dir = Path(run_dir.read_text(encoding="utf-8").strip())
    elif not run_dir.is_absolute():
        run_dir = (REPO_ROOT / run_dir).resolve()

    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    dataset_path = args.dataset
    if dataset_path is None:
        # try config path from meta
        cfg = meta.get("config_path")
        if cfg and Path(cfg).exists():
            import yaml

            raw = yaml.safe_load(Path(cfg).read_text(encoding="utf-8")) or {}
            ds = raw.get("dataset")
            if ds:
                dataset_path = REPO_ROOT / ds if not Path(ds).is_absolute() else Path(ds)
        if dataset_path is None:
            for candidate in (
                REPO_ROOT / "fixtures/datasets/hotpot_dev_slice_150.jsonl",
                REPO_ROOT / "fixtures/datasets/mini_eval.jsonl",
            ):
                if candidate.exists():
                    dataset_path = candidate
                    break

    type_map = _load_examples(dataset_path)
    per_ex = collect_per_example_rows(run_dir, type_map)
    # Enrich type_map from error files if needed
    for err in run_dir.glob("errors_*.json"):
        for row in json.loads(err.read_text(encoding="utf-8")):
            if row.get("example_id") and row.get("example_type"):
                type_map.setdefault(row["example_id"], row["example_type"])
    per_ex = collect_per_example_rows(run_dir, type_map)
    cat_rows = build_category_rows(per_ex, type_map)

    out_json = run_dir / "by_category.json"
    out_csv = run_dir / "by_category.csv"
    out_json.write_text(json.dumps(cat_rows, indent=2), encoding="utf-8")
    if cat_rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys()))
            w.writeheader()
            w.writerows(cat_rows)

    # Markdown table
    lines = [
        "# Metrics by question category",
        "",
        f"- Run: `{run_dir}`",
        f"- Dataset types from: `{dataset_path}`",
        "",
        "| Arch | Type | N | EM | F1 | soft_EM | contains_gold | soft_F1 | Abstention | Tokens | Calls |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in cat_rows:
        lines.append(
            f"| {r['architecture']} | {r['example_type']} | {r['n']} | "
            f"{r['em']} | {r['f1']} | {r['soft_em']} | {r['contains_gold']} | {r['soft_f1']} | "
            f"{r['abstention_correct']} | {r['tokens']} | {r['model_calls']} |"
        )
    md_path = run_dir / "by_category.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Pareto scatters: quality vs tokens / model_calls (per-example points)
    plot_dir = run_dir / "plots"
    for y in ("em", "f1", "contains_gold", "soft_f1"):
        plot_pareto_scatter(
            per_ex,
            plot_dir / f"pareto_{y}_vs_tokens.png",
            x_key="tokens",
            y_key=y,
            title=f"{y} vs tokens (Pareto view)",
        )
        plot_pareto_scatter(
            per_ex,
            plot_dir / f"pareto_{y}_vs_model_calls.png",
            x_key="model_calls",
            y_key=y,
            title=f"{y} vs model_calls (Pareto view)",
        )

    print(f"Wrote {out_json}, {out_csv}, {md_path}, plots under {plot_dir}")


if __name__ == "__main__":
    main()
