from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ars.models.base import ModelClient
from ars.schema import AgentRole
from ars.tracing.tracer import Tracer


class AgentMessage(BaseModel):
    role: AgentRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseAgent:
    role: AgentRole = AgentRole.SINGLE

    def __init__(self, model: ModelClient, tracer: Tracer | None = None) -> None:
        self.model = model
        self.tracer = tracer
