"""The model factory (docs/AGENTS_PLAN.md A2). No network: the HTTP transport is mocked, and so are the credentials."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx2
import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

from precursorintelligence.agents.cache import InMemoryAnswerCache
from precursorintelligence.agents.counter import InMemoryDailyCounter
from precursorintelligence.agents.model import (
    ConfiguredModel,
    GoogleTokenAuth,
    build_model,
    configured_model_name,
)
from precursorintelligence.agents.runner import AgentDefinition, DataVersion, Runtime, run_agent
from precursorintelligence.agents.settings import AgentCfg, load_agents_settings


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
    assert all(isinstance(m, ConfiguredModel) for m in model.models)
    assert all(isinstance(m.wrapped, OpenAIChatModel) for m in model.models)  # type: ignore[attr-defined]


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


# ----------------------------------------------------------------------------- the reported name vs the config id


class Answer(BaseModel):
    summary: str
    evidence_ids: list[str]


class Ask(BaseModel):
    ccn: str


def _answer_completion(reported_model: str) -> dict[str, Any]:
    """A chat completion that calls the output tool, reporting ``reported_model`` as its model."""
    call = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "final_result",
            "arguments": json.dumps({"summary": "ok", "evidence_ids": []}),
        },
    }
    out = _completion(reported_model)
    out["choices"][0]["message"] = {"role": "assistant", "content": None, "tool_calls": [call]}
    out["choices"][0]["finish_reason"] = "tool_calls"
    return out


class Vertex:
    """A fake endpoint: fails the models in ``fail``, and reports each other model as ``reports[model]``."""

    def __init__(self, reports: dict[str, str], fail: frozenset[str] = frozenset()) -> None:
        self.reports, self.fail, self.asked = reports, fail, []  # type: ignore[var-annotated]

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        model = json.loads(request.content)["model"]
        self.asked.append(model)
        if model in self.fail:
            return httpx2.Response(400, json={"error": {"message": "unavailable"}})
        return httpx2.Response(200, json=_answer_completion(self.reports[model]))


async def _fallback(inp: Ask, deps: Any) -> Answer:
    return Answer(summary="template", evidence_ids=[])


def _vertex_runtime(tmp_path, vertex: Vertex) -> Runtime:
    s = load_agents_settings()
    s.agents["demo"] = AgentCfg(max_tokens=500, temperature=0.0)
    prompt = "\n".join(["version: 1", "system: s", "instructions: i", ""])
    (tmp_path / "demo.yaml").write_text(prompt, encoding="utf-8")
    model = build_model(
        s,
        credentials=FakeCredentials(),
        transport=httpx2.MockTransport(vertex),
        transport_request=object(),
    )
    return Runtime(
        settings=s,
        model=model,
        cache=InMemoryAnswerCache(),
        counter=InMemoryDailyCounter(run_cap=10, spend_stop_usd=2.0),
        prompts_dir=tmp_path,
    )


DEMO = AgentDefinition(name="demo", input_type=Ask, output_type=Answer, fallback=_fallback)
DATA = DataVersion(v="2026-10", evidence_manifest_sha="a" * 64)


def _run(rt: Runtime):
    return asyncio.run(run_agent(rt, DEMO, {"ccn": "015009"}, data=DATA, request_id="r"))


def test_each_response_is_tagged_with_the_configured_id_and_keeps_the_reported_name():
    s = load_agents_settings()
    vertex = Vertex({s.model.primary: "gpt-oss-120b"})
    model = build_model(
        s,
        credentials=FakeCredentials(),
        transport=httpx2.MockTransport(vertex),
        transport_request=object(),
    )
    result = asyncio.run(Agent(model, output_type=Answer).run("hi"))
    assert result.response.model_name == "gpt-oss-120b"  # what Vertex said, untouched
    assert configured_model_name(result.response) == s.model.primary


def test_an_answer_is_cached_when_vertex_spells_the_primary_differently(tmp_path):
    s = load_agents_settings()
    # the endpoint reports a different string from the config id, e.g. without the "openai/" prefix
    vertex = Vertex({s.model.primary: "gpt-oss-120b"})
    rt = _vertex_runtime(tmp_path, vertex)
    first = _run(rt)
    assert not first.fallback and first.output.summary == "ok"
    assert len(rt.cache) == 1  # type: ignore[arg-type]
    again = _run(rt)
    assert again.cache_hit and vertex.asked == [s.model.primary]  # no second request
    # priced as the configured primary, not as an unknown model at the highest price
    assert first.cost_usd == pytest.approx(s.cost_usd(s.model.primary, 3, 2))


def test_an_answer_from_the_fallback_model_is_not_cached_even_if_it_reports_the_primarys_name(
    tmp_path,
):
    s = load_agents_settings()
    vertex = Vertex({s.model.fallback: s.model.primary}, fail=frozenset({s.model.primary}))
    rt = _vertex_runtime(tmp_path, vertex)
    out = _run(rt)
    assert not out.fallback and out.output.summary == "ok"
    assert vertex.asked == [s.model.primary, s.model.fallback]
    assert len(rt.cache) == 0  # type: ignore[arg-type]
