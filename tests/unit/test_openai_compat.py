"""OpenAI client records the model identifier returned by the API (no network)."""

import sys
import types
from types import SimpleNamespace

from ars.agents.planner import PlannerAgent
from ars.models.base import GenerationRequest, GenerationResponse, ModelClient
from ars.models.openai_compat import OpenAICompatClient
from ars.tracing.tracer import Tracer


class _FakeCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            id="resp-test",
            model="gpt-4o-mini-2024-07-18",
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")
            ],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        )


def test_openai_client_records_response_model(monkeypatch):
    fake_openai = types.ModuleType("openai")
    fake_openai.RateLimitError = type("RateLimitError", (Exception,), {})
    fake_openai.APIStatusError = type("APIStatusError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    client = OpenAICompatClient.__new__(OpenAICompatClient)
    client.model_name = client.name = "gpt-4o-mini"
    client.max_retries = 1
    client.response_models = set()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))

    resp = client.generate(GenerationRequest(prompt="hi"))
    assert resp.model_name == "gpt-4o-mini"
    assert resp.response_model == "gpt-4o-mini-2024-07-18"
    assert client.response_models == {"gpt-4o-mini-2024-07-18"}


def test_response_model_written_to_trace_payload():
    class _Stub(ModelClient):
        name = "stub"

        def generate(self, request):
            return GenerationResponse(
                text="plan", model_name="stub", response_model="stub-2026-01-01"
            )

    tracer = Tracer("exp", "ex1", "multi_agent")
    PlannerAgent(_Stub(), tracer).plan("Q?")
    assert tracer.events[0].payload["response_model"] == "stub-2026-01-01"
