"""Optional OpenAI-compatible client (env-gated; not required for tests)."""

from __future__ import annotations

import os
import time
from typing import Any

from ars.models.base import GenerationRequest, GenerationResponse, ModelClient
from ars.models.mock import DeterministicModel
from ars.schema import ModelSettings


class OpenAICompatClient(ModelClient):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 8,
    ) -> None:
        self.model_name = model_name
        self.name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.max_retries = max_retries
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Use DeterministicModel for offline runs."
            )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError("Install openai extra: pip install ars[openai]") from e
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        from openai import RateLimitError, APIStatusError  # type: ignore

        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
                latency = (time.perf_counter() - t0) * 1000
                choice = resp.choices[0]
                usage = getattr(resp, "usage", None)
                return GenerationResponse(
                    text=choice.message.content or "",
                    model_name=self.model_name,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_ms=latency,
                    finish_reason=choice.finish_reason or "stop",
                    raw={"id": getattr(resp, "id", None), "attempts": attempt + 1},
                )
            except RateLimitError as e:
                last_err = e
                # Honor Retry-After when present; else exponential backoff
                wait = min(2 ** attempt, 60)
                hdrs = getattr(e, "response", None)
                if hdrs is not None:
                    ra = getattr(hdrs, "headers", {}).get("retry-after") or getattr(hdrs, "headers", {}).get("Retry-After")
                    if ra:
                        try:
                            wait = max(wait, float(ra))
                        except ValueError:
                            pass
                time.sleep(wait)
            except APIStatusError as e:
                last_err = e
                if getattr(e, "status_code", None) in {500, 502, 503, 504}:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise
        assert last_err is not None
        raise last_err


def maybe_make_client(settings: ModelSettings) -> ModelClient:
    """Factory: mock by default; OpenAI when requested (no silent fallback)."""
    provider = settings.provider.lower()
    if provider in ("mock", "deterministic"):
        return DeterministicModel()
    if provider in ("openai", "openai_compat"):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "model.provider is openai but OPENAI_API_KEY is unset. "
                "Refuse silent fallback to DeterministicModel for live experiments."
            )
        return OpenAICompatClient(
            model_name=settings.model_name,
            api_key=key,
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
    raise RuntimeError(f"Unknown model provider: {settings.provider}")
