"""scripts/agents_vertex_proof.py (L4's laptop proof), run offline against a fake endpoint: no network, no cost."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from typing import Any

import httpx2

from precursorintelligence.agents import model as model_module
from precursorintelligence.agents.settings import load_agents_settings
from precursorintelligence.common.config import project_root

SCRIPT = project_root() / "scripts" / "agents_vertex_proof.py"
ANSWER = {"summary": "Recent citations.", "evidence_ids": ["score:015009:2026-10"]}


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("agents_vertex_proof", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclass looks its module up while it is defined
    spec.loader.exec_module(module)
    return module


class FakeCredentials:
    token: str | None = None

    def refresh(self, request: Any) -> None:
        self.token = "t"


def _completion(reported: str, message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": reported,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    call = {
        "id": "c",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }
    return {"role": "assistant", "content": None, "tool_calls": [call]}


def fake_vertex(request: httpx2.Request) -> httpx2.Response:
    """Calls get_score first; then answers the way the request asked: a final_answer call, or JSON text."""
    body = json.loads(request.content)
    if not any(m["role"] == "tool" for m in body["messages"]):
        message = _tool_call("get_score", {"ccn": "015009"})
    elif "response_format" in body:
        message = {"role": "assistant", "content": json.dumps(ANSWER)}
    else:
        message = _tool_call("final_answer", ANSWER)
    return httpx2.Response(200, json=_completion("gpt-oss-120b", message))


def test_the_proof_tries_both_methods_and_prints_the_reported_model_name(monkeypatch, capsys):
    real_build = model_module.build_models

    def offline_build(settings: Any, **kw: Any) -> Any:
        return real_build(
            settings,
            credentials=FakeCredentials(),
            transport=httpx2.MockTransport(fake_vertex),
            transport_request=object(),
        )

    script = _load_script()
    monkeypatch.setattr(script, "build_models", offline_build)
    assert asyncio.run(script.prove(list(script.METHODS), None)) == 0
    out = capsys.readouterr().out
    primary = load_agents_settings().model.primary
    assert "== function_calling ==" in out and "== json_schema ==" in out
    assert "reported model_name='gpt-oss-120b'" in out
    assert f"configured='{primary}'" in out
    assert "tool calls=['final_answer']" in out  # function_calling ends on the answer tool
    assert out.count("answered by the configured primary (cached): yes") == 2
    assert "works: function_calling=yes, json_schema=yes" in out
