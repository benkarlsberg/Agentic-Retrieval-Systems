#!/usr/bin/env python3
"""Recompute the paper's result tables and figures from the ARS run artifacts.

Inputs (read-only): per-example results_<arch>.json files and JSONL traces of the
two live gpt-4o-mini HotpotQA-slice runs, the slice dataset (for categories), the cached
LLM-judge verdicts judge_<arch>.json written by scripts/judge_answers.py, and the manual
audit judge_audit.csv (unconstrained run). No API calls are made here.
Outputs: analysis/results.json, analysis/tables.tex and figures/live_*.png.

Usage:
  python analysis/compute_tables.py [--ars-root ../..]  # repository root
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ARCHS = ["rag", "single_agent", "multi_agent"]
ARCH_LABEL = {"rag": "RAG", "single_agent": "Single-agent", "multi_agent": "Multi-agent"}
RUNS = {
    "unconstrained": "results/runs/live_hotpot150_unconstrained_20260915_042424",
    "equal_retrieval": "results/runs/live_hotpot150_equal_retrieval_20260915_071431",
}
DATASET = "fixtures/datasets/hotpot_dev_slice_150.jsonl"
CATEGORIES = ["multi_hop", "multi_passage", "single_hop", "unanswerable"]
# Per-example metric fields taken verbatim from the harness output.
FIELDS = [
    "em", "contains_gold", "soft_em", "f1", "soft_f1", "abstention_correct",
    "latency_ms", "tokens", "model_calls", "retrieval_calls", "unique_docs", "cost_usd",
]
N_BOOT = 10_000
SEED = 42
PAIRS = [("multi_agent", "rag"), ("single_agent", "rag"), ("multi_agent", "single_agent")]
AUDIT_FILE = "judge_audit.csv"
PAPER_DIR = Path(__file__).resolve().parents[1]


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean."""
    n = len(x)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def trace_retrieved_ids(trace: Path) -> list[str]:
    """All passage ids surfaced to the architecture, in order (union over events)."""
    ids: list[str] = []
    with trace.open(encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            p = ev.get("payload") or {}
            if ev.get("event_type") == "retrieval_result":
                ids.extend(p.get("doc_ids", []))
            elif ev.get("event_type") == "context_update" and "new_ids" in p:
                ids.extend(p["new_ids"])
    return list(dict.fromkeys(ids))


def load_judge(run_dir: Path, arch: str, ids: list[str]) -> dict[str, float]:
    """Cached LLM-judge verdicts (1.0 correct / 0.0 incorrect); every item must be judged."""
    items = json.loads((run_dir / f"judge_{arch}.json").read_text(encoding="utf-8"))["items"]
    out = {}
    for eid in ids:
        v = items[eid]["verdict"]
        if v not in ("correct", "incorrect"):
            raise SystemExit(f"{run_dir.name}/judge_{arch}.json: no verdict for {eid}")
        out[eid] = float(v == "correct")
    return out


def load_run(root: Path, run_rel: str, examples: dict[str, dict]) -> dict[str, list[dict]]:
    run_dir = root / run_rel
    out: dict[str, list[dict]] = {}
    for arch in ARCHS:
        rows = json.loads((run_dir / f"results_{arch}.json").read_text(encoding="utf-8"))
        judge = load_judge(run_dir, arch, [r["example_id"] for r in rows])
        recs = []
        for r in rows:
            ex = examples[r["example_id"]]
            m = {k: float(r["metrics"][k]) for k in FIELDS}
            gold = set(ex["gold_doc_ids"])
            titles = set(ex["metadata"].get("supporting_titles", []))
            trace = run_dir / "traces" / Path(r["trace_path"]).name
            got = trace_retrieved_ids(trace)
            if gold:
                m["evidence_recall"] = len(gold & set(got)) / len(gold)
                got_titles = {g.split("::s")[0] for g in got if g in gold}
                m["all_support_titles"] = float(titles <= got_titles) if titles else 0.0
                m["artifact_recall"] = float(r["metrics"].get("recall@k", np.nan))
            recs.append({
                "example_id": r["example_id"],
                "category": ex["example_type"],
                "answerable": ex["is_answerable"],
                "abstained": bool(r["abstained"]),
                "error": r["error"],
                "stop_reason": r["stop_reason"],
                "answer_words": len((r["answer"] or "").split()),
                "judge": judge[r["example_id"]],
                "m": m,
            })
        recs.sort(key=lambda z: z["example_id"])
        out[arch] = recs
    return out


def summarize(recs: list[dict], rng: np.random.Generator) -> dict:
    s: dict = {"n": len(recs)}
    for k in FIELDS:
        x = np.array([r["m"][k] for r in recs])
        s[k] = float(x.mean())
        if k in ("em", "contains_gold", "soft_f1"):
            s[k + "_ci95"] = boot_ci(x, rng)
    s["latency_s_median"] = float(np.median([r["m"]["latency_ms"] for r in recs]) / 1000)
    s["cost_usd_total"] = float(sum(r["m"]["cost_usd"] for r in recs))
    ans = [r for r in recs if r["answerable"]]
    una = [r for r in recs if not r["answerable"]]
    s["n_answerable"] = len(ans)
    s["em_answerable_only"] = float(np.mean([r["m"]["em"] for r in ans])) if ans else None
    s["abstain_rate_answerable"] = float(np.mean([r["abstained"] for r in ans])) if ans else None
    s["n_abstain_answerable"] = int(sum(r["abstained"] for r in ans))
    s["n_abstain_no_retrieval_answerable"] = int(
        sum(r["abstained"] and r["m"]["retrieval_calls"] == 0 for r in ans))
    s["n_abstain_unanswerable"] = int(sum(r["abstained"] for r in una))
    s["n_unanswerable"] = len(una)
    ev = [r for r in recs if "evidence_recall" in r["m"]]
    if ev:
        s["evidence_recall"] = float(np.mean([r["m"]["evidence_recall"] for r in ev]))
        s["all_support_titles"] = float(np.mean([r["m"]["all_support_titles"] for r in ev]))
        s["artifact_recall_at_k"] = float(np.mean([r["m"]["artifact_recall"] for r in ev]))
    s["mean_answer_words_non_abstained"] = float(np.mean(
        [r["answer_words"] for r in recs if not r["abstained"]] or [0]))
    s["retrieval_calls_hist"] = {str(int(k)): v for k, v in sorted(
        Counter(r["m"]["retrieval_calls"] for r in recs).items())}
    s["n_errors"] = int(sum(r["error"] is not None for r in recs))
    s["stop_reasons"] = dict(Counter(r["stop_reason"] for r in recs))
    s["n_em_ne_contains_gold"] = int(sum(r["m"]["em"] != r["m"]["contains_gold"] for r in recs))
    return s


def paired(a: list[dict], b: list[dict], key: str, rng: np.random.Generator) -> dict:
    assert [r["example_id"] for r in a] == [r["example_id"] for r in b]
    xa = np.array([r["m"][key] for r in a])
    xb = np.array([r["m"][key] for r in b])
    d = xa - xb
    lo, hi = boot_ci(d, rng)
    out = {"mean_diff": float(d.mean()), "ci95": (lo, hi), "n": len(d)}
    if key == "em":
        out["a_only_correct"] = int(((xa == 1) & (xb == 0)).sum())
        out["b_only_correct"] = int(((xa == 0) & (xb == 1)).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ars-root", default=str(PAPER_DIR.parents[1]))
    args = ap.parse_args()
    root = Path(args.ars_root)
    rng = np.random.default_rng(SEED)

    examples = {}
    with (root / DATASET).open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            examples[e["example_id"]] = e

    runs = {mode: load_run(root, rel, examples) for mode, rel in RUNS.items()}
    res: dict = {
        "config": {"n_bootstrap": N_BOOT, "seed": SEED, "ci": "percentile, 95%",
                   "runs": RUNS, "dataset": DATASET},
        "dataset": {
            "n": len(examples),
            "mean_gold_answer_words": float(np.mean(
                [len(e["gold_answer"].split()) for e in examples.values() if e["gold_answer"]])),
            "categories": dict(Counter(e["example_type"] for e in examples.values())),
            "hotpot_type_by_category": {
                c: dict(Counter(str(e["metadata"].get("hotpot_type")) for e in examples.values()
                                if e["example_type"] == c)) for c in CATEGORIES},
        },
        "overall": {}, "by_category": {}, "paired": {}, "paired_by_category": {},
        "cross_mode_agreement_em": {},
    }
    for mode, per_arch in runs.items():
        res["overall"][mode] = {a: summarize(per_arch[a], rng) for a in ARCHS}
        res["by_category"][mode] = {
            a: {c: summarize([r for r in per_arch[a] if r["category"] == c], rng) for c in CATEGORIES}
            for a in ARCHS}
        res["paired"][mode] = {}
        for x, y in [("multi_agent", "rag"), ("single_agent", "rag"), ("multi_agent", "single_agent")]:
            res["paired"][mode][f"{x}-{y}"] = {
                k: paired(per_arch[x], per_arch[y], k, rng) for k in ("em", "soft_f1")}
        res["paired_by_category"][mode] = {}
        for c in CATEGORIES:
            sel = lambda rs: [r for r in rs if r["category"] == c]  # noqa: E731
            res["paired_by_category"][mode][c] = {
                f"{x}-{y}": paired(sel(per_arch[x]), sel(per_arch[y]), "em", rng)
                for x, y in [("multi_agent", "rag"), ("single_agent", "rag")]}
    for a in ARCHS:
        u, e = runs["unconstrained"][a], runs["equal_retrieval"][a]
        assert [r["example_id"] for r in u] == [r["example_id"] for r in e]
        res["cross_mode_agreement_em"][a] = float(np.mean(
            [ru["m"]["em"] == re_["m"]["em"] for ru, re_ in zip(u, e)]))

    res["judge"] = judge_analysis(root, runs)

    out = PAPER_DIR / "analysis" / "results.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    make_figures(res)
    write_latex_tables(res)


# ------------------------------------------------------------------ LLM judge

def cohen_kappa(a: list[int], b: list[int]) -> float:
    a_, b_ = np.array(a), np.array(b)
    po = float((a_ == b_).mean())
    pa, pb = a_.mean(), b_.mean()
    pe = float(pa * pb + (1 - pa) * (1 - pb))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def judge_paired(a: list[dict], b: list[dict], rng: np.random.Generator) -> dict:
    assert [r["example_id"] for r in a] == [r["example_id"] for r in b]
    xa = np.array([r["judge"] for r in a])
    xb = np.array([r["judge"] for r in b])
    lo, hi = boot_ci(xa - xb, rng)
    return {"mean_diff": float((xa - xb).mean()), "ci95": (lo, hi), "n": len(xa),
            "a_only_correct": int(((xa == 1) & (xb == 0)).sum()),
            "b_only_correct": int(((xa == 0) & (xb == 1)).sum())}


def judge_summary(recs: list[dict], rng: np.random.Generator) -> dict:
    x = np.array([r["judge"] for r in recs])
    ans = [r for r in recs if r["answerable"]]
    s = {"n": len(recs), "acc": float(x.mean()), "acc_ci95": boot_ci(x, rng),
         "n_correct": int(x.sum()),
         "n_correct_answerable": int(sum(r["judge"] for r in ans)),
         "n_correct_unanswerable": int(sum(r["judge"] for r in recs if not r["answerable"]))}
    s["acc_answerable_only"] = s["n_correct_answerable"] / len(ans) if ans else None
    return s


def judge_analysis(root: Path, runs: dict[str, dict[str, list[dict]]]) -> dict:
    """LLM-judge accuracy, paired differences, agreement with lenient EM and the audit."""
    rng = np.random.default_rng(SEED)
    out: dict = {"overall": {}, "paired": {}, "by_category": {}, "paired_by_category": {},
                 "confusion_vs_lenient": {}, "cross_mode_agreement": {},
                 "answerable_abstentions_judged_correct": {}, "meta": {}}
    tok_in = tok_out = 0
    cost = 0.0
    models: set[str] = set()
    n_items = n_failed = 0
    prompt_versions: set[str] = set()
    for mode, rel in RUNS.items():
        out["meta"][mode] = {}
        for a in ARCHS:
            meta = json.loads((root / rel / f"judge_{a}.json").read_text(encoding="utf-8"))["meta"]
            out["meta"][mode][a] = meta
            tok_in += meta["prompt_tokens"]
            tok_out += meta["completion_tokens"]
            cost += meta["cost_usd_estimate"]
            models |= set(meta["response_models"])
            n_items += meta["n_items"]
            n_failed += meta["n_failed"]
            prompt_versions.add(meta["prompt_version"])
    out["totals"] = {"n_judgments": n_items, "n_failed": n_failed, "prompt_tokens": tok_in,
                     "completion_tokens": tok_out, "cost_usd_estimate": round(cost, 4),
                     "response_models": sorted(models), "prompt_versions": sorted(prompt_versions)}
    for mode, per_arch in runs.items():
        out["overall"][mode] = {a: judge_summary(per_arch[a], rng) for a in ARCHS}
        out["paired"][mode] = {f"{x}-{y}": judge_paired(per_arch[x], per_arch[y], rng)
                               for x, y in PAIRS}
        out["by_category"][mode] = {
            a: {c: judge_summary([r for r in per_arch[a] if r["category"] == c], rng)
                for c in CATEGORIES} for a in ARCHS}
        out["paired_by_category"][mode] = {}
        for c in CATEGORIES:
            sel = lambda rs: [r for r in rs if r["category"] == c]  # noqa: E731
            out["paired_by_category"][mode][c] = {
                f"{x}-{y}": judge_paired(sel(per_arch[x]), sel(per_arch[y]), rng)
                for x, y in PAIRS[:2]}
        out["confusion_vs_lenient"][mode] = {}
        out["answerable_abstentions_judged_correct"][mode] = {}
        for a in ARCHS:
            cm = Counter((int(r["m"]["em"]), int(r["judge"]), r["answerable"])
                         for r in per_arch[a])
            out["confusion_vs_lenient"][mode][a] = {
                "lenient1_judge1": cm[(1, 1, True)] + cm[(1, 1, False)],
                "lenient1_judge0": cm[(1, 0, True)] + cm[(1, 0, False)],
                "lenient0_judge1": cm[(0, 1, True)] + cm[(0, 1, False)],
                "lenient0_judge0": cm[(0, 0, True)] + cm[(0, 0, False)],
                "lenient0_judge1_unanswerable": cm[(0, 1, False)],
                "lenient0_judge1_answerable": cm[(0, 1, True)],
            }
            out["answerable_abstentions_judged_correct"][mode][a] = int(sum(
                r["abstained"] and r["answerable"] and r["judge"] == 1 for r in per_arch[a]))
    for a in ARCHS:
        u, e = runs["unconstrained"][a], runs["equal_retrieval"][a]
        out["cross_mode_agreement"][a] = float(np.mean(
            [ru["judge"] == re_["judge"] for ru, re_ in zip(u, e)]))
    out["audit"] = audit_analysis(root / RUNS["unconstrained"] / AUDIT_FILE,
                                  runs["unconstrained"])
    return out


def audit_analysis(path: Path, per_arch: dict[str, list[dict]]) -> dict:
    """Agreement of judge and lenient EM with the manual audit labels."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    judge_by = {(a, r["example_id"]): r["judge"] for a in ARCHS for r in per_arch[a]}
    for r in rows:
        if r["audit_verdict"] not in ("correct", "incorrect"):
            raise SystemExit(f"{path}: missing audit verdict for {r['example_id']}")
        if float(r["judge_verdict"] == "correct") != judge_by[(r["architecture"], r["example_id"])]:
            raise SystemExit(f"{path}: judge verdict out of date for {r['example_id']}")
    aud = [int(r["audit_verdict"] == "correct") for r in rows]
    jud = [int(r["judge_verdict"] == "correct") for r in rows]
    len_ = [int(r["lenient_em"]) for r in rows]
    res = {"n": len(rows),
           "judge_agree": int(sum(x == y for x, y in zip(jud, aud))),
           "lenient_agree": int(sum(x == y for x, y in zip(len_, aud))),
           "judge_kappa": cohen_kappa(jud, aud), "lenient_kappa": cohen_kappa(len_, aud),
           "by_stratum": {}}
    res["judge_agreement"] = res["judge_agree"] / res["n"]
    res["lenient_agreement"] = res["lenient_agree"] / res["n"]
    # Population sizes of each (stratum, architecture) cell among answerable questions,
    # used to reweight the stratified sample to the full unconstrained run.
    pop = Counter()
    for a in ARCHS:
        for r in per_arch[a]:
            if r["answerable"]:
                pop[("agree" if int(r["m"]["em"]) == int(r["judge"]) else "disagree", a)] += 1
    wsum = wtot = 0.0
    for st in ("disagree", "agree"):
        sel = [(j, y) for r, j, y in zip(rows, jud, aud) if r["stratum"] == st]
        res["by_stratum"][st] = {"n": len(sel), "judge_agree": int(sum(j == y for j, y in sel))}
        for a in ARCHS:
            cell = [(j, y) for r, j, y in zip(rows, jud, aud)
                    if r["stratum"] == st and r["architecture"] == a]
            if cell:
                wsum += pop[(st, a)] * np.mean([j == y for j, y in cell])
                wtot += pop[(st, a)]
    res["judge_agreement_reweighted_answerable"] = float(wsum / wtot)
    res["population_answerable"] = {f"{st}/{a}": n for (st, a), n in sorted(pop.items())}
    res["judge_audit_disagreements"] = [
        {"example_id": r["example_id"], "architecture": r["architecture"],
         "judge": r["judge_verdict"], "audit": r["audit_verdict"], "note": r["audit_note"]}
        for r, j, y in zip(rows, jud, aud) if j != y]
    return res


def f3(x: float) -> str:
    return f"{x:.3f}"


def ci(s: dict, k: str) -> str:
    lo, hi = s[k + "_ci95"]
    return f"{s[k]:.3f} {{\\scriptsize[{lo:.3f}, {hi:.3f}]}}"


def signed(x: float) -> str:
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.3f}"


def write_latex_tables(res: dict) -> None:
    """Emit LaTeX table bodies (pasted verbatim into main.tex)."""
    mode_name = {"unconstrained": "Unconstrained", "equal_retrieval": "Equal-retrieval"}
    out = []
    out.append("% ---- quality ----")
    for mode in ("unconstrained", "equal_retrieval"):
        out.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{mode_name[mode]}}}}} \\\\")
        for a in ARCHS:
            s = res["overall"][mode][a]
            out.append(f"\\quad {ARCH_LABEL[a]} & {ci(s, 'em')} & {ci(s, 'contains_gold')} & "
                       f"{ci(s, 'soft_f1')} & {f3(s['f1'])} \\\\")
        if mode == "unconstrained":
            out.append("\\midrule")
    out.append("% ---- cost ----")
    for mode in ("unconstrained", "equal_retrieval"):
        out.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{mode_name[mode]}}}}} \\\\")
        for a in ARCHS:
            s = res["overall"][mode][a]
            out.append(
                f"\\quad {ARCH_LABEL[a]} & {s['model_calls']:.2f} & {s['retrieval_calls']:.2f} & "
                f"{s['unique_docs']:.1f} & {s['tokens']:.0f} & "
                f"{s['latency_ms'] / 1000:.2f} / {s['latency_s_median']:.2f} & "
                f"{s['cost_usd_total']:.3f} \\\\")
        if mode == "unconstrained":
            out.append("\\midrule")
    out.append("% ---- category (unconstrained) ----")
    cat_label = {"multi_hop": "Multi-hop", "multi_passage": "Multi-passage", "single_hop": "Single-hop"}
    for c in ("multi_hop", "multi_passage", "single_hop"):
        row = [f"{cat_label[c]} ({res['by_category']['unconstrained']['rag'][c]['n']})"]
        for a in ARCHS:
            row.append(f3(res["by_category"]["unconstrained"][a][c]["em"]))
        for pair in ("multi_agent-rag", "single_agent-rag"):
            d = res["paired_by_category"]["unconstrained"][c][pair]
            lo, hi = d["ci95"]
            row.append(f"{signed(d['mean_diff'])} {{\\scriptsize[{signed(lo)}, {signed(hi)}]}} "
                       f"({d['a_only_correct']}/{d['b_only_correct']})")
        out.append(" & ".join(row) + " \\\\")
    out.append("% ---- diagnostics (unconstrained) ----")
    for a in ARCHS:
        s = res["overall"]["unconstrained"][a]
        out.append(
            f"{ARCH_LABEL[a]} & {s['evidence_recall']:.3f} & {s['all_support_titles']:.3f} & "
            f"{s['n_abstain_answerable']}/{s['n_answerable']} & "
            f"{s['n_abstain_unanswerable']}/{s['n_unanswerable']} & "
            f"{s['mean_answer_words_non_abstained']:.1f} \\\\")
    out.append("% ---- LLM judge (both modes) ----")
    J = res["judge"]
    for mode in ("unconstrained", "equal_retrieval"):
        out.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{mode_name[mode]}}}}} \\\\")
        for a in ARCHS:
            s = J["overall"][mode][a]
            lo, hi = s["acc_ci95"]
            cm = J["confusion_vs_lenient"][mode][a]
            if a == "rag":
                delta = "---"
            else:
                d = J["paired"][mode][f"{a}-rag"]
                dlo, dhi = d["ci95"]
                delta = (f"{signed(d['mean_diff'])} {{\\scriptsize[{signed(dlo)}, {signed(dhi)}]}} "
                         f"({d['a_only_correct']}/{d['b_only_correct']})")
            out.append(
                f"\\quad {ARCH_LABEL[a]} & {f3(res['overall'][mode][a]['em'])} & "
                f"{s['acc']:.3f} {{\\scriptsize[{lo:.3f}, {hi:.3f}]}} & {delta} & "
                f"{cm['lenient1_judge0']} / {cm['lenient0_judge1_answerable']} + "
                f"{cm['lenient0_judge1_unanswerable']} \\\\")
        if mode == "unconstrained":
            out.append("\\midrule")
    out.append("% ---- LLM judge by category (unconstrained) ----")
    cat_label_j = dict(cat_label, unanswerable="Unanswerable")
    for c in CATEGORIES:
        row = [f"{cat_label_j[c]} ({J['by_category']['unconstrained']['rag'][c]['n']})"]
        for a in ARCHS:
            row.append(f3(J["by_category"]["unconstrained"][a][c]["acc"]))
        for pair in ("multi_agent-rag", "single_agent-rag"):
            d = J["paired_by_category"]["unconstrained"][c][pair]
            lo, hi = d["ci95"]
            row.append(f"{signed(d['mean_diff'])} {{\\scriptsize[{signed(lo)}, {signed(hi)}]}} "
                       f"({d['a_only_correct']}/{d['b_only_correct']})")
        out.append(" & ".join(row) + " \\\\")
    (PAPER_DIR / "analysis" / "tables.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote analysis/tables.tex")


def make_figures(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
    })
    colors = {"rag": "#4C72B0", "single_agent": "#DD8452", "multi_agent": "#55A868"}
    markers = {"unconstrained": "o", "equal_retrieval": "s"}
    mode_label = {"unconstrained": "unconstrained", "equal_retrieval": "equal-retrieval"}
    figdir = PAPER_DIR / "figures"

    # Figure 1: quality vs. mean tokens per question (both modes).
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    for ax, key, ylab in [(axes[0], "em", "Lenient EM"), (axes[1], "soft_f1", "Soft F1")]:
        for mode in ("unconstrained", "equal_retrieval"):
            pts = []
            for a in ARCHS:
                s = res["overall"][mode][a]
                lo, hi = s[key + "_ci95"]
                ax.errorbar(s["tokens"], s[key], yerr=[[s[key] - lo], [hi - s[key]]],
                            fmt=markers[mode], color=colors[a], ms=5, capsize=2, lw=0.8,
                            mfc=colors[a] if mode == "unconstrained" else "white", mew=1.0)
                pts.append((s["tokens"], s[key]))
            # Pareto frontier (maximize quality, minimize tokens).
            pts.sort()
            front, best = [], -1.0
            for t, q in pts:
                if q > best:
                    front.append((t, q))
                    best = q
            ax.plot(*zip(*front), ls="--" if mode == "equal_retrieval" else "-",
                    color="0.55", lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks([300, 500, 1000, 2000, 3000])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(250, 3600)
        ax.set_xlabel("Mean tokens per question (log scale)")
        ax.set_ylabel(ylab)
        ax.grid(True, which="major", lw=0.3, alpha=0.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=colors[a], label=ARCH_LABEL[a]) for a in ARCHS]
    handles += [Line2D([], [], marker=markers[m], ls="-" if m == "unconstrained" else "--",
                       color="0.4", mfc="0.4" if m == "unconstrained" else "white",
                       label=mode_label[m]) for m in markers]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figdir / "live_pareto_tokens.png")
    plt.close(fig)

    # Figure 2: per-category lenient EM by architecture (unconstrained).
    cats = ["multi_hop", "multi_passage", "single_hop"]
    cat_label = {"multi_hop": "multi-hop", "multi_passage": "multi-passage", "single_hop": "single-hop"}
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    w = 0.26
    xs = np.arange(len(cats))
    for i, a in enumerate(ARCHS):
        vals, los, his = [], [], []
        for c in cats:
            s = res["by_category"]["unconstrained"][a][c]
            vals.append(s["em"])
            los.append(s["em"] - s["em_ci95"][0])
            his.append(s["em_ci95"][1] - s["em"])
        ax.bar(xs + (i - 1) * w, vals, w, color=colors[a], label=ARCH_LABEL[a],
               yerr=[los, his], capsize=2, error_kw={"lw": 0.7}, edgecolor="white", lw=0.5)
    n = {c: res["by_category"]["unconstrained"]["rag"][c]["n"] for c in cats}
    ax.set_xticks(xs, [f"{cat_label[c]}\n($n$={n[c]})" for c in cats])
    ax.set_ylabel("Lenient EM")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(figdir / "live_category_em.png")
    plt.close(fig)
    print("wrote figures/live_pareto_tokens.png, figures/live_category_em.png")


if __name__ == "__main__":
    main()
