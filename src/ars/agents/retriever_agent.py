from __future__ import annotations

from ars.agents.base import AgentMessage, BaseAgent
from ars.retrieval.base import Retriever
from ars.retrieval.rerank import SimpleReranker
from ars.schema import AgentRole, RetrievedDoc
from ars.tracing.events import TraceEventType


class RetrieverAgent(BaseAgent):
    """Role-specialized retriever (fact / evidence / contrast)."""

    role = AgentRole.RETRIEVER

    def __init__(
        self,
        model,  # unused for retrieval itself but kept for interface uniformity
        retriever: Retriever,
        specialty: str = "general",
        tracer=None,
        use_rerank: bool = False,
    ) -> None:
        super().__init__(model, tracer)
        self.retriever = retriever
        self.specialty = specialty
        self.use_rerank = use_rerank
        self.reranker = SimpleReranker() if use_rerank else None

    def retrieve(self, query: str, top_k: int = 5, exclude_ids: set[str] | None = None) -> list[RetrievedDoc]:
        # Specialty hint appended lightly for dense/bm25 cueing
        q = query
        if self.specialty == "evidence":
            q = f"{query} evidence facts"
        elif self.specialty == "contrast":
            q = f"{query} comparison alternative"
        if self.tracer:
            self.tracer.emit(
                TraceEventType.RETRIEVAL_QUERY,
                payload={"query": q, "specialty": self.specialty, "top_k": top_k},
                agent_role=self.role.value,
            )
        hits = self.retriever.retrieve_with_ids(q, top_k=top_k, exclude_ids=exclude_ids)
        if self.reranker:
            hits = self.reranker.rerank(query, hits, top_k=top_k)
        if self.tracer:
            self.tracer.emit(
                TraceEventType.RETRIEVAL_RESULT,
                payload={
                    "doc_ids": [h.doc_id for h in hits],
                    "scores": [h.score for h in hits],
                    "specialty": self.specialty,
                },
                agent_role=self.role.value,
            )
        return hits

    def as_message(self, hits: list[RetrievedDoc]) -> AgentMessage:
        lines = [f"[{h.doc_id}] title: {h.title}\n{h.text}" for h in hits]
        return AgentMessage(
            role=self.role,
            content="\n\n".join(lines),
            metadata={"doc_ids": [h.doc_id for h in hits], "specialty": self.specialty},
        )
