"""The model factory (docs/AGENTS_PLAN.md A2). No network: the HTTP transport is mocked, and so are the credentials."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx2
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from precursorintelligence.agents.cache import InMemoryAnswerCache
from precursorintelligence.agents.counter import InMemoryDailyCounter
from precursorintelligence.agents.errors import ModelCallFailed
from precursorintelligence.agents.model import (
    LANGSMITH_TRACING_VARS,
    GoogleTokenAuth,
    ModelAnswer,
    ModelPair,
    build_models,
    disable_langsmith_tracing,
)
from precursorintelligence.agents.runner import AgentDefinition, DataVersion, Runtime, run_agent
from precursorintelligence.agents.settings import AgentCfg, load_agents_settings

FALLBACK_MODEL = "openai/gpt-oss-20b-maas"


@pytest.fixture
def with_fallback(monkeypatch):
    """A fallback model configured through the environment (the repo default has none)."""
    monkeypatch.setenv("PI_AGENTS_FALLBACK_MODEL", FALLBACK_MODEL)


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


def _ask(pair: ModelPair) -> ModelAnswer:
    return asyncio.run(pair.ainvoke([HumanMessage("hello")], lambda model: model))


def test_by_default_the_factory_builds_the_primary_alone():
    pair = build_models(
        load_agents_settings(), credentials=FakeCredentials(), transport_request=object()
    )
    assert isinstance(pair.primary, ChatOpenAI)
    assert (pair.fallback, pair.fallback_id) == (None, None)


def test_the_openai_client_makes_no_retries_of_its_own():
    # the runner has its own fallback; a hidden retry would double a slow call and go uncounted
    pair = build_models(
        load_agents_settings(), credentials=FakeCredentials(), transport_request=object()
    )
    assert isinstance(pair.primary, ChatOpenAI)
    assert pair.primary.root_async_client.max_retries == 0  # type: ignore[union-attr]


def test_with_a_failing_model_exactly_one_request_is_sent():
    s = load_agents_settings()
    rec = Recorder(fail_models=frozenset({s.model.primary}))
    pair = build_models(
        s,
        credentials=FakeCredentials(),
        transport=httpx2.MockTransport(rec),
        transport_request=object(),
    )
    with pytest.raises(ModelCallFailed):
        _ask(pair)
    assert len(rec.requests) == 1


@pytest.mark.parametrize("var", LANGSMITH_TRACING_VARS)
def test_langsmith_tracing_is_forced_off_even_when_the_environment_turns_it_on(monkeypatch, var):
    import langsmith.utils
    from langchain_core.tracers.context import _tracing_v2_is_enabled

    monkeypatch.setenv(var, "true")
    disable_langsmith_tracing()
    assert all(os.environ[name] == "false" for name in LANGSMITH_TRACING_VARS)
    assert not langsmith.utils.tracing_is_enabled()
    assert not _tracing_v2_is_enabled()  # what LangChain asks before sending a run to LangSmith


def test_building_the_models_turns_langsmith_off(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    build_models(load_agents_settings(), credentials=FakeCredentials(), transport_request=object())
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_with_no_fallback_model_a_failing_call_fails():
    s = load_agents_settings()
    rec = Recorder(fail_models=frozenset({s.model.primary}))
    pair = build_models(
        s,
        credentials=FakeCredentials(),
        transport=httpx2.MockTransport(rec),
        transport_request=object(),
    )
    with pytest.raises(ModelCallFailed) as exc:
        _ask(pair)
    assert s.model.primary in str(exc.value) and "then" not in str(exc.value)
    assert len(rec.requests) == 1


def test_the_factory_builds_the_primary_with_the_fallback_behind_it(with_fallback):
    s = load_agents_settings()
    pair = build_models(s, credentials=FakeCredentials(), transport_request=object())
    assert (pair.primary_id, pair.fallback_id) == (s.model.primary, s.model.fallback)
    assert isinstance(pair.primary, ChatOpenAI) and isinstance(pair.fallback, ChatOpenAI)
    assert [pair.primary.model_name, pair.fallback.model_name] == [
        s.model.primary,
        s.model.fallback,
    ]


def test_every_request_gets_a_freshly_refreshed_token():
    s = load_agents_settings()
    creds, rec = FakeCredentials(), Recorder()
    pair = build_models(
        s, credentials=creds, transport=httpx2.MockTransport(rec), transport_request=object()
    )
    _ask(pair)
    _ask(pair)
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


def test_a_failing_primary_falls_back_to_the_fallback_model(with_fallback):
    s = load_agents_settings()
    rec = Recorder(fail_models=frozenset({s.model.primary}))
    pair = build_models(
        s,
        credentials=FakeCredentials(),
        transport=httpx2.MockTransport(rec),
        transport_request=object(),
    )
    reply = _ask(pair)
    assert reply.message.content == "ok"
    asked = [json.loads(r.content)["model"] for r in rec.requests]
    assert asked == [s.model.primary, s.model.fallback]
    assert reply.configured_id == s.model.fallback


def test_when_both_models_fail_the_call_fails_with_both_errors(with_fallback):
    s = load_agents_settings()
    rec = Recorder(fail_models=frozenset({s.model.primary, s.model.fallback}))
    pair = build_models(
        s,
        credentials=FakeCredentials(),
        transport=httpx2.MockTransport(rec),
        transport_request=object(),
    )
    with pytest.raises(ModelCallFailed) as exc:
        _ask(pair)
    assert s.model.primary in str(exc.value) and s.model.fallback in str(exc.value)


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
            "name": "final_answer",
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
    model = build_models(
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


def test_each_reply_carries_the_configured_id_and_keeps_the_reported_name():
    s = load_agents_settings()
    vertex = Vertex({s.model.primary: "gpt-oss-120b"})
    pair = build_models(
        s,
        credentials=FakeCredentials(),
        transport=httpx2.MockTransport(vertex),
        transport_request=object(),
    )
    reply = _ask(pair)
    assert reply.reported_name == "gpt-oss-120b"  # what Vertex said, untouched
    assert reply.configured_id == s.model.primary


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
    tmp_path, with_fallback
):
    s = load_agents_settings()
    vertex = Vertex({s.model.fallback: s.model.primary}, fail=frozenset({s.model.primary}))
    rt = _vertex_runtime(tmp_path, vertex)
    out = _run(rt)
    assert not out.fallback and out.output.summary == "ok"
    assert vertex.asked == [s.model.primary, s.model.fallback]
    assert len(rt.cache) == 0  # type: ignore[arg-type]
