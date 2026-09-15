"""Provider-neutral structured trace events (JSONL)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TraceEventType(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    AGENT_MESSAGE = "agent_message"
    PROMPT = "prompt"
    MODEL_CALL = "model_call"
    MODEL_RESPONSE = "model_response"
    ROUTING = "routing"
    SUBQUESTION = "subquestion"
    RETRIEVAL_QUERY = "retrieval_query"
    RETRIEVAL_RESULT = "retrieval_result"
    CONTEXT_UPDATE = "context_update"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    BUDGET_CHECK = "budget_check"
    TERMINATION = "termination"
    ANSWER = "answer"
    CITATION = "citation"
    ERROR = "error"
    METRIC = "metric"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class TraceEvent(BaseModel):
    event_type: TraceEventType
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    experiment_id: str
    example_id: str
    architecture: str
    step: int = 0
    agent_role: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    prompt_hash: str | None = None
    latency_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    error: str | None = None

    def to_jsonl(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)
