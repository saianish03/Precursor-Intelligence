"""agents/vertex.py and scripts/vertex_smoke.py (L4): the endpoint, the gpt-oss profile and the auth, all over a
mocked transport. Nothing here reaches the network or Google."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest

from precursorintelligence.agents import vertex
from precursorintelligence.agents.vertex import (
    CLOUD_PLATFORM_SCOPE,
    DEFAULT_MODEL,
    gpt_oss_profile,
    vertex_base_url,
    vertex_model,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "vertex_smoke.py"


def load_script() -> Any:
    spec = importlib.util.spec_from_file_location("vertex_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["vertex_smoke"] = module  # pydantic needs the module to resolve the script's models
    spec.loader.exec_module(module)
    return module


class FakeCredentials:
    """Starts without a token; each refresh issues a new one."""

    def __init__(self) -> None:
        self.token: str | None = None
        self.refreshes = 0
        self.expired = True

    @property
    def valid(self) -> bool:
        return self.token is not None and not self.expired

    def refresh(self, request: Any) -> None:
        self.refreshes += 1
        self.token = f"tok-{self.refreshes}"
        self.expired = False


def _completion(tool_name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{call_id}",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "openai/gpt-oss-120b-maas",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340},
    }


class FakeVertex:
    """Plays gpt-oss on the Chat Completions endpoint: lookup_tag first, then the output tool."""

    def __init__(self, passage_id: str) -> None:
        self.passage_id = passage_id
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if len(self.requests) == 1:
            body = _completion("lookup_tag", {"tag": "F0689"}, "call_1")
        else:
            args = {
                "tag": "F0689",
                "summary": "Keep residents safe.",
                "evidence_ids": [self.passage_id],
            }
            body = _completion("final_result", args, "call_2")
        return httpx2.Response(200, json=body)


@pytest.mark.parametrize(
    ("region", "url"),
    [
        (
            "global",
            "https://aiplatform.googleapis.com/v1/projects/p1/locations/global/endpoints/openapi",
        ),
        (
            "us-central1",
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p1/locations/us-central1/endpoints/openapi",
        ),
    ],
)
def test_the_base_url_uses_the_global_host_or_the_regional_one(region: str, url: str) -> None:
    assert vertex_base_url("p1", region) == url


def test_the_factory_builds_the_model_without_any_network_or_token() -> None:
    creds = FakeCredentials()
    model = vertex_model("p1", "global", DEFAULT_MODEL, credentials=creds)
    assert model.model_name == "openai/gpt-oss-120b-maas"
    assert str(model.client.base_url).rstrip("/") == vertex_base_url("p1", "global")
    assert creds.refreshes == 0


def test_without_credentials_it_asks_adc_for_the_cloud_platform_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_default(scopes: list[str] | None = None) -> tuple[FakeCredentials, str]:
        seen["scopes"] = scopes
        return FakeCredentials(), "p1"

    import google.auth

    monkeypatch.setattr(google.auth, "default", fake_default)
    vertex_model("p1")
    assert seen["scopes"] == [CLOUD_PLATFORM_SCOPE]


def test_the_gpt_oss_profile_never_forces_a_tool_and_uses_tool_output() -> None:
    profile = gpt_oss_profile(DEFAULT_MODEL)
    assert profile["supports_forced_tool_choice"] is False
    assert profile["supports_json_schema_output"] is False
    assert profile["default_structured_output_mode"] == "tool"
    model = vertex_model("p1", credentials=FakeCredentials())
    assert model.profile["supports_forced_tool_choice"] is False


def test_a_run_over_the_wire_sends_auto_tool_choice_and_a_fresh_bearer_token() -> None:
    smoke = load_script()
    fake = FakeVertex(smoke.PASSAGE_ID)
    creds = FakeCredentials()
    model = vertex_model(
        "p1",
        "global",
        DEFAULT_MODEL,
        credentials=creds,
        transport=httpx2.MockTransport(fake),
        transport_request=object(),
    )
    report = asyncio.run(smoke.smoke(model, DEFAULT_MODEL))

    assert report["passed"], report
    assert report["tool_calls"] == ["lookup_tag", "final_result"]
    assert (report["input_tokens"], report["output_tokens"]) == (600, 80)
    assert report["estimated_cost_usd"] == pytest.approx((600 * 0.15 + 80 * 0.60) / 1e6)
    assert len(fake.requests) == 2
    for request in fake.requests:
        assert str(request.url) == (
            "https://aiplatform.googleapis.com/v1/projects/p1/locations/global/endpoints/openapi"
            "/chat/completions"
        )
        assert request.headers["Authorization"] == "Bearer tok-1"
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-oss-120b-maas"
        # the default OpenAI profile sends "required" for the output tool, which gpt-oss rejects
        assert body["tool_choice"] == "auto"
        assert "response_format" not in body  # no json_schema output on Vertex open models
        assert {t["function"]["name"] for t in body["tools"]} >= {"lookup_tag", "final_result"}
    # a token is reused while valid and refreshed once it expires: the credentials, never a token string
    assert creds.refreshes == 1


def test_an_expired_token_is_refreshed_before_the_request() -> None:
    creds = FakeCredentials()
    auth = vertex.GoogleTokenAuth(creds, transport_request=object())
    request = httpx2.Request("POST", "https://example.invalid")
    next(auth.sync_auth_flow(request))
    assert request.headers["Authorization"] == "Bearer tok-1"
    creds.expired = True
    request = httpx2.Request("POST", "https://example.invalid")
    next(auth.sync_auth_flow(request))
    assert request.headers["Authorization"] == "Bearer tok-2"


def test_the_smoke_script_dry_run_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert load_script().main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "'tool_called': True" in out


def test_the_smoke_script_needs_a_project_for_a_real_run() -> None:
    with pytest.raises(SystemExit):
        load_script().main(["--region", "global"])


def test_the_usage_limits_are_d4s() -> None:
    limits = load_script().LIMITS
    assert (limits.request_limit, limits.input_tokens_limit, limits.output_tokens_limit) == (
        6,
        200_000,
        12_000,
    )
