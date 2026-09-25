from __future__ import annotations

import re

from ars.agents.base import AgentMessage, BaseAgent
from ars.models.base import GenerationRequest
from ars.schema import AgentRole, Citation
from ars.tracing.events import TraceEventType


class SynthesizerAgent(BaseAgent):
    role = AgentRole.SYNTHESIZER

    def synthesize(self, question: str, context: str) -> tuple[str, list[Citation], AgentMessage]:
        prompt = (
            f"Question: {question}\n"
            f"Context:\n{context}\n"
            "Answer using only the context. If unsupported, ABSTAIN. Include CITATIONS with doc ids."
        )
        resp = self.model.generate(
            GenerationRequest(
                prompt=prompt,
                system="You are a careful answer synthesizer.",
                metadata={"role": "synthesizer", "task": "answer"},
            )
        )
        answer, citations = self._parse(resp.text)
        if self.tracer:
            self.tracer.emit(
                TraceEventType.ANSWER,
                payload={
                    "answer": answer,
                    "citations": [c.model_dump() for c in citations],
                    **self._response_meta(resp),
                },
                agent_role=self.role.value,
                prompt=prompt,
                tokens_in=resp.prompt_tokens,
                tokens_out=resp.completion_tokens,
                latency_ms=resp.latency_ms,
            )
        msg = AgentMessage(
            role=self.role,
            content=resp.text,
            metadata={"tokens": resp.prompt_tokens + resp.completion_tokens},
        )
        return answer, citations, msg

    def _parse(self, text: str) -> tuple[str, list[Citation]]:
        citations: list[Citation] = []
        m = re.search(r"CITATIONS:\s*(.+)$", text, re.I | re.M)
        body = text
        if m:
            ids = re.findall(r"doc_[a-z0-9_]+", m.group(1), re.I)
            citations = [Citation(doc_id=i) for i in ids]
            body = text[: m.start()].strip()
        abstain = body.upper().startswith("ABSTAIN")
        return body, citations if not abstain or citations else citations
