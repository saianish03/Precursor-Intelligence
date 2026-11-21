"""The model factory (docs/AGENTS_PLAN.md A2). No network: the HTTP transport is mocked, and so are the credentials."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx2
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

from precursorintelligence.agents.model import GoogleTokenAuth, build_model
from precursorintelligence.agents.settings import load_agents_settings


class FakeCredentials:
    """Hands out a new token on each refresh, like a token that keeps expiring."""

    def __init__(self) -> None:
        self.token: str | None = None
        self.refreshes = 0

    def refresh(self, request: Any) -> None:
        self.refreshes += 1
        self.token = f"token-{self.refreshes}"


def _completion(model: str, content: str = "ok") -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


class Recorder:
    def __init__(self, fail_models: frozenset[str] = frozenset()) -> None:
        self.requests: list[httpx2.Request] = []
        self.fail_models = fail_models

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        model = json.loads(request.content)["model"]
        if model in self.fail_models:
            return httpx2.Response(400, json={"error": {"message": "unavailable"}})
        return httpx2.Response(200, json=_completion(model))


def test_the_factory_builds_the_primary_with_the_fallback_behind_it():
    s = load_agents_settings()
    model = build_model(s, credentials=FakeCredentials(), transport_request=object())
    assert isinstance(model, FallbackModel)
    names = [m.model_name for m in model.models]
    assert names == [s.model.primary, s.model.fallback]
    assert all(isinstance(m, OpenAIChatModel) for m in model.models)


def test_every_request_gets_a_freshly_refreshed_token():
    s = load_agents_settings()
    creds, rec = FakeCredentials(), Recorder()
    model = build_model(
        s, credentials=creds, transport=httpx2.MockTransport(rec), transport_request=object()
    )
    agent = Agent(model)

    async def two_runs() -> None:
        await agent.run("one")
        await agent.run("two")

    asyncio.run(two_runs())
    assert creds.refreshes == 2
    assert [r.headers["Authorization"] for r in rec.requests] == [
        "Bearer token-1",
        "Bearer token-2",
    ]
    assert all(str(r.url).startswith(s.model.base_url) for r in rec.requests)


def test_the_auth_keeps_no_token_string():
    creds = FakeCredentials()
    auth = GoogleTokenAuth(creds, transport_request=object())
    flow = auth.sync_auth_flow(httpx2.Request("POST", "https://example.com"))
    sent = next(flow)
    assert sent.headers["Authorization"] == "Bearer token-1"
    # nothing on the auth object is the token: only the credentials (which own it) and the transport
    for value in vars(auth).values():
        assert not isinstance(value, str)
    # a later token change is picked up by the next request, so no copy of the old one was kept
    flow = auth.sync_auth_flow(httpx2.Request("POST", "https://example.com"))
    assert next(flow).headers["Authorization"] == "Bearer token-2"


def test_a_failing_primary_falls_back_to_the_fallback_model():
    s = load_agents_settings()
    rec = Recorder(fail_models=frozenset({s.model.primary}))
    model = build_model(
        s,
        credentials=FakeCredentials(),
        transport=httpx2.MockTransport(rec),
        transport_request=object(),
    )
    result = asyncio.run(Agent(model).run("hello"))
    assert result.output == "ok"
    asked = [json.loads(r.content)["model"] for r in rec.requests]
    assert asked == [s.model.primary, s.model.fallback]
    assert result.response.model_name == s.model.fallback
