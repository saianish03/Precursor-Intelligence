"""L4's laptop proof (docs/AGENTS_PLAN.md L4): one Pydantic AI run with a tool call and structured output against
Vertex AI's OpenAI-compatible endpoint, through the same model factory the API uses.

It prints, for every model response, the **model name Vertex reports** next to the configured id that answered. The
runner caches an answer only when the configured primary wrote it, decided from which model object answered (the
``configured_model`` tag), so a differently spelled reported name is fine; this shows what that name actually is.

Run it by hand, never in CI. It makes real, billed calls (about a cent):

    gcloud auth application-default login
    uv run --extra agents python scripts/agents_vertex_proof.py            # the primary, falling back to the 20b
    uv run --extra agents python scripts/agents_vertex_proof.py --only fallback

Model ids, the project and the region come from configs/agents_runtime.yaml and its environment variables
(PI_AGENTS_MODEL, PI_AGENTS_FALLBACK_MODEL, PI_GCP_PROJECT, PI_AGENTS_REGION).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.fallback import FallbackModel

from precursorintelligence.agents.model import build_model, configured_model_name
from precursorintelligence.agents.settings import load_agents_settings


class Score(BaseModel):
    evidence_id: str
    p: float


class Answer(BaseModel):
    summary: str
    evidence_ids: list[str]


async def prove(only: str | None) -> int:
    s = load_agents_settings()
    model = build_model(s)
    if only is not None:
        assert isinstance(model, FallbackModel)
        model = model.models[0 if only == "primary" else 1]
    agent: Agent[None, Answer] = Agent(
        model,
        output_type=Answer,
        instructions="Call get_score once for the CCN, then answer in one sentence, citing its evidence_id.",
    )

    @agent.tool_plain
    def get_score(ccn: str) -> Score:
        """The facility's score."""
        return Score(evidence_id=f"score:{ccn}:2026-10", p=0.31)

    print(f"endpoint:   {s.model.base_url}")
    print(f"configured: primary={s.model.primary} fallback={s.model.fallback}")
    result = await agent.run("Why is facility 015009 risky?")

    responses = [m for m in result.all_messages() if isinstance(m, ModelResponse)]
    for i, r in enumerate(responses, 1):
        tools = [p.tool_name for p in r.parts if isinstance(p, ToolCallPart)]
        print(
            f"response {i}: reported model_name={r.model_name!r} "
            f"configured={configured_model_name(r)!r} tool calls={tools}"
        )
    usage = result.usage
    print(f"output:     {json.dumps(result.output.model_dump())}")
    print(
        f"usage:      {usage.requests} requests, {usage.input_tokens} input and {usage.output_tokens} output tokens"
    )

    called_tool = any(
        isinstance(p, ToolCallPart) and p.tool_name == "get_score"
        for r in responses
        for p in r.parts
    )
    cited = "score:015009:2026-10" in result.output.evidence_ids
    by_primary = all(configured_model_name(r) == s.model.primary for r in responses)
    print(
        f"tool call: {'yes' if called_tool else 'NO'}; cited the tool's ID: {'yes' if cited else 'NO'}"
    )
    print(f"answered by the configured primary (would be cached): {'yes' if by_primary else 'no'}")
    return 0 if called_tool and cited else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only",
        choices=["primary", "fallback"],
        help="call one model alone instead of the primary with the fallback behind it",
    )
    sys.exit(asyncio.run(prove(parser.parse_args().only)))


if __name__ == "__main__":
    main()
