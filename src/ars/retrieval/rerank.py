"""Simple lexical overlap reranker (offline)."""

from __future__ import annotations

import re

from ars.schema import RetrievedDoc


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class SimpleReranker:
    name = "lexical_overlap"

    def rerank(self, query: str, docs: list[RetrievedDoc], top_k: int | None = None) -> list[RetrievedDoc]:
        q = _tokens(query)
        scored: list[tuple[float, RetrievedDoc]] = []
        for d in docs:
            dt = _tokens(f"{d.title} {d.text}")
            overlap = len(q & dt) / max(len(q), 1)
            scored.append((overlap + 0.01 * d.score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        k = top_k or len(scored)
        out: list[RetrievedDoc] = []
        for rank, (score, d) in enumerate(scored[:k]):
            out.append(
                RetrievedDoc(
                    doc_id=d.doc_id,
                    score=float(score),
                    title=d.title,
                    text=d.text,
                    rank=rank,
                )
            )
        return out
