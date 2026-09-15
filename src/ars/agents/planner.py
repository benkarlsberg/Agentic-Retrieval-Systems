from __future__ import annotations

from ars.agents.base import AgentMessage, BaseAgent
from ars.models.base import GenerationRequest
from ars.schema import AgentRole
from ars.tracing.events import TraceEventType


class PlannerAgent(BaseAgent):
    role = AgentRole.PLANNER

    def plan(self, question: str) -> AgentMessage:
        prompt = (
            f"Question: {question}\n"
            "Decompose into subquestions and choose a retrieval strategy."
        )
        resp = self.model.generate(
            GenerationRequest(
                prompt=prompt,
                system="You are a retrieval planner.",
                metadata={"role": "planner", "task": "plan"},
            )
        )
        if self.tracer:
            self.tracer.emit(
                TraceEventType.AGENT_MESSAGE,
                payload={"content": resp.text},
                agent_role=self.role.value,
                prompt=prompt,
                tokens_in=resp.prompt_tokens,
                tokens_out=resp.completion_tokens,
                latency_ms=resp.latency_ms,
            )
            self.tracer.emit(
                TraceEventType.SUBQUESTION,
                payload={"plan": resp.text},
                agent_role=self.role.value,
            )
        return AgentMessage(role=self.role, content=resp.text, metadata={"tokens": resp.prompt_tokens + resp.completion_tokens})
