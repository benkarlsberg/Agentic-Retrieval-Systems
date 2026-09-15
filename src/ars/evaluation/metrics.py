"""Evaluation metrics: EM, token F1, recall@k, MRR, citations, abstention, etc."""

from __future__ import annotations

import re
import string
from collections import Counter

from ars.schema import Example, ExampleResult


def normalize_answer(s: str | None) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    # remove abstain prefix for comparison of answerable cases handled separately
    s = re.sub(r"\bcitations:.*", "", s, flags=re.I | re.S)
    s = s.replace("abstain:", "").strip()
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str, aliases: list[str] | None = None) -> float:
    np_ = normalize_answer(pred)
    candidates = [gold] + (aliases or [])
    for c in candidates:
        nc = normalize_answer(c)
        if not nc:
            continue
        if np_ == nc or nc in np_ or np_ in nc:
            return 1.0
    return 0.0


def lexical_similarity(pred: str, gold: str) -> float:
    """Fallback semantic similarity via token Jaccard."""
    p = set(normalize_answer(pred).split())
    g = set(normalize_answer(gold).split())
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    return len(p & g) / len(p | g)


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int | None = None) -> float:
    if not gold_ids:
        return 0.0  # undefined; caller may skip
    ids = retrieved_ids[:k] if k else retrieved_ids
    hit = len(set(ids) & set(gold_ids))
    return hit / len(set(gold_ids))


def mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return 0.0
    gold = set(gold_ids)
    for i, did in enumerate(retrieved_ids):
        if did in gold:
            return 1.0 / (i + 1)
    return 0.0


def citation_precision_recall(
    cited_ids: list[str], gold_ids: list[str]
) -> tuple[float, float]:
    if not cited_ids and not gold_ids:
        return 1.0, 1.0
    if not cited_ids:
        return 0.0, 0.0 if gold_ids else 1.0
    if not gold_ids:
        return 0.0, 1.0
    cited = set(cited_ids)
    gold = set(gold_ids)
    tp = len(cited & gold)
    prec = tp / len(cited)
    rec = tp / len(gold)
    return prec, rec


def abstention_correctness(abstained: bool, is_answerable: bool) -> float:
    # Correct if (unanswerable and abstained) or (answerable and not abstained)
    if not is_answerable:
        return 1.0 if abstained else 0.0
    return 1.0 if not abstained else 0.0


def compute_example_metrics(
    result: ExampleResult,
    example: Example,
    retrieved_ids: list[str] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    gold = example.gold_answer or ""
    pred = result.answer

    if example.is_answerable and gold:
        metrics["em"] = exact_match(pred, gold, example.aliases)
        metrics["f1"] = token_f1(pred, gold)
        metrics["lexical_sim"] = lexical_similarity(pred, gold)
    else:
        metrics["em"] = 0.0
        metrics["f1"] = 0.0
        metrics["lexical_sim"] = 0.0

    metrics["abstention_correct"] = abstention_correctness(
        result.abstained, example.is_answerable
    )

    cited = [c.doc_id for c in result.citations]
    if example.gold_doc_ids:
        prec, rec = citation_precision_recall(cited, example.gold_doc_ids)
        metrics["citation_precision"] = prec
        metrics["citation_recall"] = rec
        if retrieved_ids is not None:
            metrics["recall@k"] = recall_at_k(retrieved_ids, example.gold_doc_ids)
            metrics["mrr"] = mrr(retrieved_ids, example.gold_doc_ids)
        else:
            # fall back to unique docs order unknown — use citation set as proxy list
            metrics["recall@k"] = recall_at_k(cited, example.gold_doc_ids)
            metrics["mrr"] = mrr(cited, example.gold_doc_ids)

    metrics["latency_ms"] = result.latency_ms
    metrics["model_calls"] = float(result.model_calls)
    metrics["retrieval_calls"] = float(result.retrieval_calls)
    metrics["unique_docs"] = float(result.unique_docs)
    metrics["duplicate_docs"] = float(result.duplicate_docs)
    metrics["tokens"] = float(result.prompt_tokens + result.completion_tokens)
    metrics["cost_usd"] = result.estimated_cost_usd
    if "coordination_messages" in result.metrics:
        metrics["coordination_messages"] = result.metrics["coordination_messages"]
    return metrics
