#!/usr/bin/env python3
"""Generate markdown/HTML report artifacts from a run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ars.reporting.trace_viewer import write_trace_html  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    if run_dir.name == "latest" and run_dir.is_file():
        run_dir = Path(run_dir.read_text(encoding="utf-8").strip())
    elif not run_dir.is_absolute():
        run_dir = (REPO_ROOT / run_dir).resolve()

    meta_path = run_dir / "meta.json"
    comparison = run_dir / "comparison.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    rows = json.loads(comparison.read_text(encoding="utf-8")) if comparison.exists() else []

    lines = [
        "# Experiment Report",
        "",
        "> **LABEL: offline-fixture results** — DeterministicModel / fixture corpus.",
        "> Do not interpret as live LLM production performance.",
        "",
        f"- Experiment: `{meta.get('experiment_id', '?')}`",
        f"- Comparison mode: `{meta.get('comparison_mode', '?')}`",
        f"- N examples: {meta.get('n_examples', '?')}",
        f"- Architectures: {', '.join(meta.get('architectures', []))}",
        "",
        "## Aggregate metrics",
        "",
        "| Architecture | EM | F1 | Abstention | Latency (ms) | Model calls | Retrieval calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('architecture')} | {r.get('em')} | {r.get('f1')} | "
            f"{r.get('abstention_correct')} | {r.get('latency_ms')} | "
            f"{r.get('model_calls')} | {r.get('retrieval_calls')} |"
        )
    lines += ["", "## Artifacts", f"- Run dir: `{run_dir}`", ""]
    report_path = run_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    traces = list((run_dir / "traces").glob("*.jsonl")) if (run_dir / "traces").exists() else []
    for t in traces[:5]:
        write_trace_html(t, t.with_suffix(".html"))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
