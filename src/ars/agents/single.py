"""Single-agent iterative controller helpers."""

from __future__ import annotations

from ars.agents.base import BaseAgent
from ars.models.base import GenerationRequest
from ars.routing.actions import Action, parse_action
from ars.schema import AgentRole
from ars.tracing.events import TraceEventType


class SingleAgentController(BaseAgent):
    role = AgentRole.SINGLE

    def decide(self, question: str, context: str, step: int) -> Action:
        prompt = (
            f"Question: {question}\n"
            f"Step: {step}\n"
            f"Context:\n{context}\n"
            "Decide ACTION: retrieve | reformulate | stop | abstain. "
            "If retrieve/reformulate, include QUERY."
        )
        resp = self.model.generate(
            GenerationRequest(
                prompt=prompt,
                system="You are a bounded retrieval agent.",
                metadata={"role": "single", "task": "decide"},
            )
        )
        action = parse_action(resp.text)
        if self.tracer:
            self.tracer.emit(
                TraceEventType.ROUTING,
                payload={
                    "raw": resp.text,
                    "action": action.model_dump(),
                    **self._response_meta(resp),
                },
                agent_role=self.role.value,
                prompt=prompt,
                tokens_in=resp.prompt_tokens,
                tokens_out=resp.completion_tokens,
                latency_ms=resp.latency_ms,
            )
        action.reason = (action.reason or "") + f"|tokens={resp.prompt_tokens + resp.completion_tokens}"
        return action
