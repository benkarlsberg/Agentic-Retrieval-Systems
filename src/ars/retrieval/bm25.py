"""BM25 sparse retrieval via rank_bm25 (with pure-Python fallback)."""

from __future__ import annotations

import re
from collections import Counter
from math import log

from ars.retrieval.base import Retriever
from ars.schema import Document, RetrievedDoc


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _PureBM25:
    """Minimal BM25 Okapi if rank_bm25 is unavailable."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        df: Counter[str] = Counter()
        for doc in corpus_tokens:
            for t in set(doc):
                df[t] += 1
        self.idf = {
            t: log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.N
        for i, doc in enumerate(self.corpus):
            tf = Counter(doc)
            dl = self.doc_len[i]
            s = 0.0
            for term in query:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                s += idf * (freq * (self.k1 + 1)) / (
                    freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            scores[i] = s
        return scores


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self) -> None:
        self.documents: list[Document] = []
        self._bm25: object | None = None
        self._tokens: list[list[str]] = []

    def index(self, documents: list[Document]) -> None:
        self.documents = list(documents)
        self._tokens = [_tokenize(f"{d.title} {d.text}") for d in self.documents]
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._tokens)
        except ImportError:
            self._bm25 = _PureBM25(self._tokens)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        if not self.documents or self._bm25 is None:
            return []
        q = _tokenize(query)
        scores = list(self._bm25.get_scores(q))  # type: ignore[attr-defined]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out: list[RetrievedDoc] = []
        for rank, idx in enumerate(ranked):
            d = self.documents[idx]
            out.append(
                RetrievedDoc(
                    doc_id=d.doc_id,
                    score=float(scores[idx]),
                    title=d.title,
                    text=d.text,
                    rank=rank,
                )
            )
        return out
