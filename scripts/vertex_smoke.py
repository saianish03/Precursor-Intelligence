"""L4's day-one laptop proof (docs/AGENTS_PLAN.md L4): one Pydantic AI run on gpt-oss on Vertex AI that calls a
tool and returns structured output. Runbook: docs/runbooks/vertex-l4.md.

The real run makes billed calls (well under a cent: D4's limits cap it at about $0.04). Run it by hand, never in CI:

    gcloud auth application-default login
    uv run --extra agents python scripts/vertex_smoke.py --project <PROJECT_ID>
    uv run --extra agents python scripts/vertex_smoke.py --project <PROJECT_ID> --model openai/gpt-oss-20b-maas
    uv run --extra agents python scripts/vertex_smoke.py --project <PROJECT_ID> --region us-central1

``--dry-run`` runs the same agent on Pydantic AI's FunctionModel (scripted like gpt-oss: a tool call, then the
output tool) with the same gpt-oss profile: no network, no credentials, no cost.

It checks that ``lookup_tag`` was called and that the output validated as ``TagFinding`` citing the passage ID the
tool returned, prints the model id, latency, token usage and an estimated cost, and exits 0 on a pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import UsageLimits

from precursorintelligence.agents.vertex import (
    DEFAULT_MODEL,
    DEFAULT_REGION,
    gpt_oss_profile,
    vertex_base_url,
    vertex_model,
)

TAG = "F0689"
PASSAGE_ID = f"passage:cms_tag:{TAG}:2021-01"
PASSAGE = (
    "F0689 Free of Accident Hazards/Supervision/Devices: the facility must ensure that the resident environment "
    "remains as free of accident hazards as is possible, and that each resident receives adequate supervision "
    "and assistance devices to prevent accidents."
)

# USD per million tokens (input, output): Vertex launch prices, as in §7 and configs/agents_runtime.yaml (#104).
# Check https://cloud.google.com/vertex-ai/generative-ai/pricing before relying on them.
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b-maas": (0.15, 0.60),
    "openai/gpt-oss-20b-maas": (0.075, 0.30),
}

# D4's starting values for one run, plus a tool-call cap; max_tokens as for Agent 2 (evidence retriever).
LIMITS = UsageLimits(
    request_limit=6,
    input_tokens_limit=200_000,
    output_tokens_limit=12_000,
    tool_calls_limit=4,
)
SETTINGS = ModelSettings(max_tokens=2000, temperature=0.0)

INSTRUCTIONS = (
    "You explain CMS deficiency tags. Call lookup_tag exactly once for the tag in the question, then answer "
    "with a one-sentence summary of the passage and cite the evidence_id the tool returned."
)


class Passage(BaseModel):
    evidence_id: str
    tag: str
    text: str


class TagFinding(BaseModel):
    tag: str = Field(description="the deficiency tag, e.g. F0689")
    summary: str = Field(min_length=1, description="one sentence")
    evidence_ids: list[str] = Field(min_length=1, description="evidence IDs returned by lookup_tag")


def build_agent(model: Model) -> Agent[None, TagFinding]:
    agent: Agent[None, TagFinding] = Agent(
        model, output_type=TagFinding, instructions=INSTRUCTIONS, model_settings=SETTINGS
    )

    @agent.tool_plain
    def lookup_tag(tag: str) -> Passage:
        """The regulation passage for a CMS deficiency tag such as F0689."""
        return Passage(evidence_id=PASSAGE_ID, tag=TAG, text=PASSAGE)

    return agent


def _scripted_gpt_oss(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """What a well-behaved gpt-oss run looks like: the tool first, then the output tool."""
    called = any(
        isinstance(p, ToolCallPart) and p.tool_name == "lookup_tag"
        for m in messages
        if isinstance(m, ModelResponse)
        for p in m.parts
    )
    if not called:
        return ModelResponse(parts=[ToolCallPart("lookup_tag", {"tag": TAG})])
    output_tool = info.output_tools[0].name
    args = {
        "tag": TAG,
        "summary": "F0689 requires a hazard-free environment.",
        "evidence_ids": [PASSAGE_ID],
    }
    return ModelResponse(parts=[ToolCallPart(output_tool, args)])


def dry_run_model(model_id: str) -> FunctionModel:
    return FunctionModel(
        _scripted_gpt_oss, model_name=f"dry-run:{model_id}", profile=gpt_oss_profile(model_id)
    )


def estimated_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """At the listed price, or the highest listed price for an unknown model, so it never comes out low."""
    price_in, price_out = PRICES.get(
        model_id, (max(p[0] for p in PRICES.values()), max(p[1] for p in PRICES.values()))
    )
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


async def smoke(model: Model, model_id: str) -> dict[str, Any]:
    """One run; returns the facts to print, with ``passed``."""
    agent = build_agent(model)
    start = time.perf_counter()
    result = await agent.run(f"What does deficiency tag {TAG} require?", usage_limits=LIMITS)
    latency_s = time.perf_counter() - start

    responses = [m for m in result.all_messages() if isinstance(m, ModelResponse)]
    tool_calls = [p.tool_name for r in responses for p in r.parts if isinstance(p, ToolCallPart)]
    output = result.output
    usage = result.usage
    checks = {
        "tool_called": "lookup_tag" in tool_calls,
        "output_validated": isinstance(output, TagFinding),
        "cited_tool_id": PASSAGE_ID in output.evidence_ids,
    }
    return {
        "model": model_id,
        "reported_model_names": sorted({r.model_name for r in responses if r.model_name}),
        "latency_s": round(latency_s, 2),
        "requests": usage.requests,
        "tool_calls": tool_calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "estimated_cost_usd": round(
            estimated_cost_usd(model_id, usage.input_tokens, usage.output_tokens), 6
        ),
        "output": output.model_dump(),
        "checks": checks,
        "passed": all(checks.values()),
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project", help="GCP project id (required unless --dry-run)")
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="'global' (default; the location Google lists for gpt-oss) or a region such as us-central1",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Vertex model id (default {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="run offline on FunctionModel: no network, no cost"
    )
    args = parser.parse_args(argv)
    if not args.dry_run and not args.project:
        parser.error("--project is required unless --dry-run")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        model: Model = dry_run_model(args.model)
        print("mode:     dry run (FunctionModel, no network)")
    else:
        model = vertex_model(args.project, args.region, args.model)
        print(f"endpoint: {vertex_base_url(args.project, args.region)}")
    try:
        report = asyncio.run(smoke(model, args.model))
    except Exception as exc:  # the proof's job is to show the failure, not a traceback
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"model:    {report['model']} (reported {report['reported_model_names']})")
    print(f"latency:  {report['latency_s']} s over {report['requests']} requests")
    print(f"tools:    {report['tool_calls']}")
    print(f"usage:    {report['input_tokens']} input, {report['output_tokens']} output tokens")
    print(f"cost:     ~${report['estimated_cost_usd']:.6f} (estimate)")
    print(f"checks:   {report['checks']}")
    print(json.dumps(report))
    print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
