"""Hybrid BM25 + dense fusion (RRF)."""

from __future__ import annotations

from ars.retrieval.base import Retriever
from ars.retrieval.bm25 import BM25Retriever
from ars.retrieval.dense import DenseRetriever
from ars.schema import Document, RetrievedDoc


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(
        self,
        sparse: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.sparse = sparse or BM25Retriever()
        self.dense = dense or DenseRetriever()
        self.rrf_k = rrf_k
        self.documents: list[Document] = []

    def index(self, documents: list[Document]) -> None:
        self.documents = list(documents)
        self.sparse.index(documents)
        self.dense.index(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        s_hits = self.sparse.retrieve(query, top_k=top_k * 2)
        d_hits = self.dense.retrieve(query, top_k=top_k * 2)
        scores: dict[str, float] = {}
        docs: dict[str, RetrievedDoc] = {}
        for rank, h in enumerate(s_hits):
            scores[h.doc_id] = scores.get(h.doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            docs[h.doc_id] = h
        for rank, h in enumerate(d_hits):
            scores[h.doc_id] = scores.get(h.doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            docs[h.doc_id] = h
        ranked_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_k]
        out: list[RetrievedDoc] = []
        for rank, did in enumerate(ranked_ids):
            base = docs[did]
            out.append(
                RetrievedDoc(
                    doc_id=did,
                    score=scores[did],
                    title=base.title,
                    text=base.text,
                    rank=rank,
                )
            )
        return out
