"""Retriever interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ars.schema import Document, RetrievedDoc


class Retriever(ABC):
    name: str = "base"

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        raise NotImplementedError

    def retrieve_with_ids(
        self, query: str, top_k: int = 5, exclude_ids: set[str] | None = None
    ) -> list[RetrievedDoc]:
        hits = self.retrieve(query, top_k=top_k * 2 if exclude_ids else top_k)
        if exclude_ids:
            hits = [h for h in hits if h.doc_id not in exclude_ids][:top_k]
        return hits
