#!/usr/bin/env python3
"""Grade saved answers from the live HotpotQA-slice runs with an LLM judge.

For every (question, gold answer, system answer) in the given run directories the
judge model returns {"verdict": "correct" | "incorrect", "reason": "..."}. The judge
is not told which architecture produced an answer, and requests are issued in a
seeded random order across runs and architectures. No pipeline is re-run: only the
answers already stored in results_<arch>.json are graded.

Results are cached per run directory in judge_<arch>.json, keyed by example id and a
hash of (prompt version, judge model, question, gold, answer), so a rerun only calls
the API for missing or failed items.

Usage (needs OPENAI_API_KEY and the openai package; see pyproject extra "openai"):
  python scripts/judge_answers.py                  # both live HotpotQA runs
  python scripts/judge_answers.py --dry-run        # report what would be sent
  python scripts/judge_answers.py audit-sample     # draw the manual-audit sample
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "fixtures" / "datasets" / "hotpot_dev_slice_150.jsonl"
DEFAULT_RUNS = [
    "results/runs/live_hotpot150_unconstrained_20260915_042424",
    "results/runs/live_hotpot150_equal_retrieval_20260915_071431",
]
ARCHS = ["rag", "single_agent", "multi_agent"]

JUDGE_MODEL = "gpt-4o-mini"
PROMPT_VERSION = "judge-v2"
# USD per 1M tokens (gpt-4o-mini list price), used only for the logged cost estimate.
PRICE_IN, PRICE_OUT = 0.15, 0.60
SHUFFLE_SEED = 20260925
AUDIT_SEED = 7

_COMMON_HEAD = """You grade an answer produced by a question-answering system. The system answer may be a full sentence; judge the answer it commits to, not its wording."""

_JSON_TAIL = """Respond with a single JSON object and nothing else:
{"reason": "<one short sentence naming the answer the system committed to, if any>", "verdict": "correct" or "incorrect"}"""

# Answerable questions: the judge sees only the reference-matching rules.
SYSTEM_PROMPT_ANSWERABLE = _COMMON_HEAD + """

The question is answerable and the reference answer is the ground truth. Do not use outside knowledge to overrule it.

Mark CORRECT only if the system answer clearly commits to the reference entity or value as its answer to the question. Accept aliases, alternate spellings, shorter or longer forms of a name that clearly identify the same entity, formatting differences (for example of dates or numbers), extra accurate detail, and reasonable paraphrases. For yes/no references, the answer must clearly commit to the same yes/no value, stated or unambiguously implied.

Mark INCORRECT if the system answer does any of the following:
- abstains, or says the answer cannot be determined or is not in the provided information (always incorrect here, because the question is answerable);
- hedges between two or more candidate answers without committing to the reference;
- names a different entity or value;
- mentions the reference string only incidentally while asserting a different answer to the question.

""" + _JSON_TAIL

# Unanswerable questions: the correct behaviour is to abstain.
SYSTEM_PROMPT_UNANSWERABLE = _COMMON_HEAD + """

This question is unanswerable: it has no answer in the source material. Mark CORRECT if and only if the system answer abstains or states that the information is not available. Any substantive answer, including a guess, is INCORRECT.

""" + _JSON_TAIL


def build_messages(question: str, gold: list[str] | None, answer: str) -> list[dict[str, str]]:
    system = SYSTEM_PROMPT_ANSWERABLE if gold else SYSTEM_PROMPT_UNANSWERABLE
    return [{"role": "system", "content": system},
            {"role": "user", "content": build_user_prompt(question, gold, answer)}]


def build_user_prompt(question: str, gold: list[str] | None, answer: str) -> str:
    if not gold:
        return f"Question: {question}\nReference answer: none (unanswerable)\nSystem answer: {answer}"
    ref = gold[0] if len(gold) == 1 else " | ".join(gold) + "  (any of these)"
    return f"Question: {question}\nReference answer: {ref}\nSystem answer: {answer}"


def parse_verdict(text: str) -> dict[str, str]:
    """Parse the judge's reply; raise ValueError unless it is the exact JSON schema."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):] if "{" in s else s
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("judge reply is not a JSON object")
    verdict = obj.get("verdict")
    if not isinstance(verdict, str) or verdict.strip().lower() not in ("correct", "incorrect"):
        raise ValueError(f"invalid verdict: {verdict!r}")
    reason = obj.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("reason must be a string")
    return {"verdict": verdict.strip().lower(), "reason": reason.strip()}


def item_key(question: str, gold: list[str] | None, answer: str, model: str = JUDGE_MODEL,
             prompt_version: str = PROMPT_VERSION) -> str:
    payload = json.dumps([prompt_version, model, question, gold or None, answer],
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * PRICE_IN + completion_tokens * PRICE_OUT) / 1e6


# ---------------------------------------------------------------- cache handling

def cache_path(run_dir: Path, arch: str) -> Path:
    return run_dir / f"judge_{arch}.json"


def load_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"meta": {}, "items": {}}


def cached_ok(cache: dict[str, Any], example_id: str, key: str) -> bool:
    it = cache["items"].get(example_id)
    return bool(it) and it.get("key") == key and it.get("verdict") in ("correct", "incorrect")


def summarize_cache(cache: dict[str, Any]) -> dict[str, Any]:
    items = list(cache["items"].values())
    ok = [i for i in items if i.get("verdict") in ("correct", "incorrect")]
    return {
        "judge_model_requested": JUDGE_MODEL,
        "prompt_version": PROMPT_VERSION,
        "temperature": 0,
        "n_items": len(items),
        "n_correct": sum(i["verdict"] == "correct" for i in ok),
        "n_failed": len(items) - len(ok),
        "response_models": sorted({i["response_model"] for i in ok if i.get("response_model")}),
        "prompt_tokens": sum(i.get("prompt_tokens", 0) for i in items),
        "completion_tokens": sum(i.get("completion_tokens", 0) for i in items),
        "cost_usd_estimate": round(sum(i.get("cost_usd", 0.0) for i in items), 6),
        "price_usd_per_1m_tokens": {"input": PRICE_IN, "output": PRICE_OUT},
    }


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    cache["meta"] = summarize_cache(cache)
    cache["items"] = dict(sorted(cache["items"].items()))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- data

def load_examples(path: Path = DATASET) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                out[e["example_id"]] = e
    return out


def gold_list(ex: dict) -> list[str] | None:
    if not ex.get("is_answerable", True) or not ex.get("gold_answer"):
        return None
    return [ex["gold_answer"]] + [a for a in ex.get("aliases") or [] if a != ex["gold_answer"]]


def build_jobs(run_dirs: list[Path], examples: dict[str, dict]) -> list[dict]:
    jobs = []
    for run_dir in run_dirs:
        for arch in ARCHS:
            rows = json.loads((run_dir / f"results_{arch}.json").read_text(encoding="utf-8"))
            for r in rows:
                ex = examples[r["example_id"]]
                gold = gold_list(ex)
                answer = r["answer"] or ""
                jobs.append({
                    "run_dir": run_dir, "arch": arch, "example_id": r["example_id"],
                    "question": ex["question"], "gold": gold, "answer": answer,
                    "key": item_key(ex["question"], gold, answer),
                })
    return jobs


# ---------------------------------------------------------------- API

def make_openai_caller(model: str = JUDGE_MODEL) -> Callable[[list[dict]], dict]:
    """Return a function messages -> {text, response_model, prompt_tokens, completion_tokens}."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set; refusing to run the judge.")
    from openai import OpenAI  # imported lazily so tests and --dry-run need no openai

    client = OpenAI(max_retries=0, timeout=60)

    def call(messages: list[dict]) -> dict:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
            messages=messages,
        )
        usage = resp.usage
        return {
            "text": resp.choices[0].message.content or "",
            "response_model": resp.model,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }

    return call


def judge_one(job: dict, call: Callable[[list[dict]], dict], max_attempts: int = 6,
              sleep: Callable[[float], None] = time.sleep) -> dict:
    """Grade one job with retries on transient errors and on malformed replies."""
    messages = build_messages(job["question"], job["gold"], job["answer"])
    ptok = ctok = 0
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            out = call(messages)
            ptok += out["prompt_tokens"]
            ctok += out["completion_tokens"]
            parsed = parse_verdict(out["text"])
            return {
                "key": job["key"], "verdict": parsed["verdict"], "reason": parsed["reason"],
                "judge_model_requested": JUDGE_MODEL, "response_model": out["response_model"],
                "prompt_version": PROMPT_VERSION, "prompt_tokens": ptok,
                "completion_tokens": ctok, "cost_usd": cost_usd(ptok, ctok),
                "attempts": attempt, "raw": out["text"],
                "judged_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            }
        except Exception as e:  # noqa: BLE001 - network, rate limit, or parse error
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_attempts:
                sleep(min(2 ** attempt, 30))
    return {
        "key": job["key"], "verdict": None, "error": last_err, "attempts": max_attempts,
        "judge_model_requested": JUDGE_MODEL, "prompt_version": PROMPT_VERSION,
        "prompt_tokens": ptok, "completion_tokens": ctok, "cost_usd": cost_usd(ptok, ctok),
    }


def run_judge(run_dirs: list[Path], call: Callable[[list[dict]], dict] | None, workers: int = 8,
              dry_run: bool = False, examples: dict[str, dict] | None = None,
              max_attempts: int = 6, sleep: Callable[[float], None] = time.sleep) -> dict:
    examples = examples if examples is not None else load_examples()
    jobs = build_jobs(run_dirs, examples)
    caches = {(j["run_dir"], j["arch"]): None for j in jobs}
    for k in caches:
        caches[k] = load_cache(cache_path(*k))
    todo = [j for j in jobs if not cached_ok(caches[(j["run_dir"], j["arch"])],
                                             j["example_id"], j["key"])]
    random.Random(SHUFFLE_SEED).shuffle(todo)  # blind order across runs and architectures
    print(f"{len(jobs)} judgments total, {len(jobs) - len(todo)} cached, {len(todo)} to run")
    if dry_run or not todo:
        return {"total": len(jobs), "todo": len(todo)}
    assert call is not None
    lock = threading.Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(judge_one, j, call, max_attempts, sleep): j for j in todo}
        for fut in as_completed(futs):
            j = futs[fut]
            res = fut.result()
            with lock:
                caches[(j["run_dir"], j["arch"])]["items"][j["example_id"]] = res
                done += 1
                if done % 100 == 0:
                    for k, c in caches.items():
                        save_cache(cache_path(*k), c)
                    print(f"  {done}/{len(todo)}")
    failed = 0
    for k, c in caches.items():
        save_cache(cache_path(*k), c)
        failed += c["meta"]["n_failed"]
        m = c["meta"]
        print(f"{k[0].name} {k[1]}: {m['n_correct']}/{m['n_items']} correct, "
              f"{m['n_failed']} failed, {m['prompt_tokens']}+{m['completion_tokens']} tokens, "
              f"${m['cost_usd_estimate']:.4f}, models={m['response_models']}")
    return {"total": len(jobs), "todo": len(todo), "failed": failed}


# ---------------------------------------------------------------- audit sample

AUDIT_FIELDS = ["example_id", "architecture", "category", "question", "gold", "answer",
                "lenient_em", "judge_verdict", "judge_reason", "stratum",
                "audit_verdict", "audit_note"]


def draw_audit_sample(run_dir: Path, examples: dict[str, dict], n_disagree: int = 30,
                      n_agree: int = 30, seed: int = AUDIT_SEED) -> list[dict]:
    """Stratified sample of answerable items: judge/lenient disagreements and agreements,
    spread evenly over architectures. Unanswerable items are excluded because lenient EM
    is 0 on them by definition, so every judge-credited abstention is a disagreement."""
    pools: dict[tuple[str, str], list[dict]] = {}
    for arch in ARCHS:
        rows = {r["example_id"]: r for r in
                json.loads((run_dir / f"results_{arch}.json").read_text(encoding="utf-8"))}
        cache = load_cache(cache_path(run_dir, arch))
        for eid, it in cache["items"].items():
            if it.get("verdict") not in ("correct", "incorrect"):
                continue
            r, ex = rows[eid], examples[eid]
            if not gold_list(ex):
                continue
            lenient = int(r["metrics"]["em"])
            judge = int(it["verdict"] == "correct")
            stratum = "disagree" if lenient != judge else "agree"
            pools.setdefault((stratum, arch), []).append({
                "example_id": eid, "architecture": arch, "category": ex["example_type"],
                "question": ex["question"], "gold": ex.get("gold_answer") or "[unanswerable]",
                "answer": r["answer"] or "", "lenient_em": lenient,
                "judge_verdict": it["verdict"], "judge_reason": it.get("reason", ""),
                "stratum": stratum, "audit_verdict": "", "audit_note": "",
            })
    rng = random.Random(seed)
    sample = []
    for stratum, total in (("disagree", n_disagree), ("agree", n_agree)):
        groups = {a: sorted(pools.get((stratum, a), []), key=lambda z: z["example_id"])
                  for a in ARCHS}
        for g in groups.values():
            rng.shuffle(g)
        # Round-robin over architectures so the sample is spread across them.
        picked = []
        while len(picked) < total and any(groups.values()):
            for a in ARCHS:
                if groups[a] and len(picked) < total:
                    picked.append(groups[a].pop())
        sample.extend(picked)
    return sample


def write_audit_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", nargs="?", default="judge", choices=["judge", "audit-sample"])
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--audit-out", default=str(REPO_ROOT / DEFAULT_RUNS[0] / "judge_audit.csv"))
    args = ap.parse_args(argv)
    run_dirs = [(REPO_ROOT / r).resolve() if not Path(r).is_absolute() else Path(r)
                for r in args.runs]
    examples = load_examples()
    if args.command == "audit-sample":
        out = Path(args.audit_out)
        if out.exists():
            print(f"{out} exists; not overwriting (it may contain audit labels)")
            return 1
        rows = draw_audit_sample(run_dirs[0], examples)
        write_audit_csv(out, rows)
        print(f"wrote {len(rows)} rows to {out}")
        return 0
    call = None if args.dry_run else make_openai_caller()
    res = run_judge(run_dirs, call, workers=args.workers, dry_run=args.dry_run,
                    examples=examples)
    return 1 if res.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
