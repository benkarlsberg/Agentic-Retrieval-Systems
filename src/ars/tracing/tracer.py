"""JSONL tracer for structured experiment traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ars.tracing.events import TraceEvent, TraceEventType, hash_text


class Tracer:
    def __init__(
        self,
        experiment_id: str,
        example_id: str,
        architecture: str,
        path: Path | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.example_id = example_id
        self.architecture = architecture
        self.path = path
        self.events: list[TraceEvent] = []
        self.step = 0
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        event_type: TraceEventType,
        payload: dict[str, Any] | None = None,
        agent_role: str | None = None,
        prompt: str | None = None,
        latency_ms: float | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cost_usd: float | None = None,
        error: str | None = None,
        advance_step: bool = False,
    ) -> TraceEvent:
        if advance_step:
            self.step += 1
        event = TraceEvent(
            event_type=event_type,
            experiment_id=self.experiment_id,
            example_id=self.example_id,
            architecture=self.architecture,
            step=self.step,
            agent_role=agent_role,
            payload=payload or {},
            prompt_hash=hash_text(prompt) if prompt else None,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            error=error,
        )
        self.events.append(event)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(event.to_jsonl() + "\n")
        return event

    def flush(self) -> list[dict[str, Any]]:
        return [e.model_dump() for e in self.events]
