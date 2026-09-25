"""Provider-neutral model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    prompt: str
    system: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    stop: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationResponse(BaseModel):
    text: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str = "stop"
    # Model identifier reported by the provider in its response (e.g. a dated
    # snapshot behind an alias). None for offline models.
    response_model: str | None = None


class ModelClient(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError

    def estimate_tokens(self, text: str) -> int:
        # Rough heuristic: ~4 chars per token
        return max(1, len(text) // 4)
