"""scripts/agents_vertex_proof.py (L4's laptop proof), run offline against a fake endpoint: no network, no cost."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import httpx2

from precursorintelligence.agents import model as model_module
from precursorintelligence.agents.settings import load_agents_settings
from precursorintelligence.common.config import project_root

SCRIPT = project_root() / "scripts" / "agents_vertex_proof.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("agents_vertex_proof", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCredentials:
    token: str | None = None

    def refresh(self, request: Any) -> None:
        self.token = "t"


def _completion(reported: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    call = {
        "id": "c",
        "type": "function",
        "function": {"name": tool, "arguments": json.dumps(args)},
    }
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": reported,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_the_proof_prints_the_reported_model_name(monkeypatch, capsys):
    replies = [
        _completion("gpt-oss-120b", "get_score", {"ccn": "015009"}),
        _completion(
            "gpt-oss-120b",
            "final_result",
            {"summary": "Recent citations.", "evidence_ids": ["score:015009:2026-10"]},
        ),
    ]

    def endpoint(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=replies.pop(0))

    real_build = model_module.build_model

    def offline_build(settings: Any, **kw: Any) -> Any:
        return real_build(
            settings,
            credentials=FakeCredentials(),
            transport=httpx2.MockTransport(endpoint),
            transport_request=object(),
        )

    script = _load_script()
    monkeypatch.setattr(script, "build_model", offline_build)
    assert asyncio.run(script.prove(None)) == 0
    out = capsys.readouterr().out
    primary = load_agents_settings().model.primary
    assert "reported model_name='gpt-oss-120b'" in out
    assert f"configured='{primary}'" in out
    assert "answered by the configured primary (would be cached): yes" in out
    assert Path(SCRIPT).is_file()
