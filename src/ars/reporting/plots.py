from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_metric_bars(
    summaries: list[Any],
    metric: str,
    path: Path,
    title: str | None = None,
) -> Path | None:
    """Generate a bar chart; returns None if matplotlib unavailable."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = []
    values = []
    lows = []
    highs = []
    for s in summaries:
        labels.append(s.architecture.value if hasattr(s.architecture, "value") else str(s.architecture))
        values.append(s.aggregate_metrics.get(metric, 0.0))
        lows.append(s.aggregate_metrics.get(f"{metric}_ci_low", values[-1]))
        highs.append(s.aggregate_metrics.get(f"{metric}_ci_high", values[-1]))
    yerr = [
        [max(0.0, v - lo) for v, lo in zip(values, lows, strict=True)],
        [max(0.0, hi - v) for v, hi in zip(values, highs, strict=True)],
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, yerr=yerr, capsize=4, color=["#4C78A8", "#F58518", "#54A24B"][: len(labels)])
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by architecture (offline fixture)")
    ax.set_ylim(0, max(1.0, max(values + highs) * 1.15) if values else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_budget_tradeoff(
    rows: list[dict[str, Any]],
    path: Path,
    x_key: str = "metric_tokens",
    y_key: str = "metric_f1",
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    by_arch: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        arch = r.get("architecture", "?")
        by_arch.setdefault(arch, []).append((float(r.get(x_key, 0)), float(r.get(y_key, 0))))
    for arch, pts in by_arch.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, label=arch, alpha=0.7)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title("Budget–quality tradeoff (offline fixture)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_pareto_scatter(
    rows: list[dict[str, Any]],
    path: Path,
    x_key: str,
    y_key: str,
    title: str | None = None,
    group_key: str = "architecture",
) -> Path | None:
    """Scatter quality (y) vs cost (x); useful for Pareto-style inspection."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    by_g: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        g = str(r.get(group_key, "?"))
        by_g.setdefault(g, []).append((float(r.get(x_key, 0) or 0), float(r.get(y_key, 0) or 0)))
    for g, pts in by_g.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, label=g, alpha=0.65, s=28)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title or f"{y_key} vs {x_key}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
