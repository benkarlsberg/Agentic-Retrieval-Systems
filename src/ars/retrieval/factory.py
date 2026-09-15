from __future__ import annotations

from ars.retrieval.base import Retriever
from ars.retrieval.bm25 import BM25Retriever
from ars.retrieval.dense import DenseRetriever
from ars.retrieval.hybrid import HybridRetriever
from ars.schema import Document


def make_retriever(mode: str, documents: list[Document]) -> Retriever:
    mode = mode.lower()
    if mode == "bm25":
        r: Retriever = BM25Retriever()
    elif mode == "dense":
        r = DenseRetriever()
    elif mode == "hybrid":
        r = HybridRetriever()
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    r.index(documents)
    return r
