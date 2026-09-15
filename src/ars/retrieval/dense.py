"""Dense retrieval with fixture/stub embeddings for offline use."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from ars.retrieval.base import Retriever
from ars.schema import Document, RetrievedDoc


class FixtureEmbedder:
    """Deterministic bag-of-words hashing embedder (offline, no model download)."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float64)
            tokens = re.findall(r"[a-z0-9]+", t.lower())
            for tok in tokens:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
                v[idx] += sign
            n = np.linalg.norm(v)
            if n > 0:
                v /= n
            vecs.append(v)
        return np.stack(vecs, axis=0)

    def save(self, path: Path, ids: list[str], matrix: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), matrix)
        path.with_suffix(".json").write_text(json.dumps(ids), encoding="utf-8")

    def load(self, path: Path) -> tuple[list[str], np.ndarray]:
        ids = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        matrix = np.load(path.with_suffix(".npy"))
        return ids, matrix


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, embedder: FixtureEmbedder | None = None) -> None:
        self.embedder = embedder or FixtureEmbedder()
        self.documents: list[Document] = []
        self._matrix: np.ndarray | None = None

    def index(self, documents: list[Document]) -> None:
        self.documents = list(documents)
        texts = [f"{d.title} {d.text}" for d in documents]
        self._matrix = self.embedder.embed(texts)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        if self._matrix is None or not self.documents:
            return []
        q = self.embedder.embed([query])[0]
        # cosine similarity (vectors already normalized)
        scores = self._matrix @ q
        ranked = np.argsort(-scores)[:top_k]
        out: list[RetrievedDoc] = []
        for rank, idx in enumerate(ranked):
            i = int(idx)
            d = self.documents[i]
            out.append(
                RetrievedDoc(
                    doc_id=d.doc_id,
                    score=float(scores[i]),
                    title=d.title,
                    text=d.text,
                    rank=rank,
                )
            )
        return out
