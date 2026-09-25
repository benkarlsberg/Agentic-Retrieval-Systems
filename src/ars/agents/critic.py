from __future__ import annotations

from ars.agents.base import AgentMessage, BaseAgent
from ars.models.base import GenerationRequest
from ars.schema import AgentRole
from ars.tracing.events import TraceEventType


class CriticAgent(BaseAgent):
    role = AgentRole.CRITIC

    def critique(self, question: str, context: str) -> AgentMessage:
        prompt = (
            f"Question: {question}\n"
            f"Context:\n{context}\n"
            "Judge if evidence is sufficient to answer, insufficient, or should abstain."
        )
        resp = self.model.generate(
            GenerationRequest(
                prompt=prompt,
                system="You are a strict evidence critic.",
                metadata={"role": "critic", "task": "critique"},
            )
        )
        if self.tracer:
            self.tracer.emit(
                TraceEventType.AGENT_MESSAGE,
                payload={"content": resp.text, **self._response_meta(resp)},
                agent_role=self.role.value,
                prompt=prompt,
                tokens_in=resp.prompt_tokens,
                tokens_out=resp.completion_tokens,
                latency_ms=resp.latency_ms,
            )
        return AgentMessage(
            role=self.role,
            content=resp.text,
            metadata={"tokens": resp.prompt_tokens + resp.completion_tokens},
        )
