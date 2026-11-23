"""The shared output models (docs/AGENTS_PLAN.md A5) and how run_agent stamps them. No paid model, no network."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Annotated, Any

import pytest
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, ValidationError

from precursorintelligence.agents.cache import InMemoryAnswerCache
from precursorintelligence.agents.counter import InMemoryDailyCounter
from precursorintelligence.agents.model import ModelPair
from precursorintelligence.agents.outputs import (
    CAVEAT,
    AgentOutput,
    Driver,
    Section,
    evidence_ids_of,
)
from precursorintelligence.agents.runner import (
    ANSWER_TOOL,
    AgentDefinition,
    DataVersion,
    RunDeps,
    Runtime,
    build_cache_key,
    run_agent,
)
from precursorintelligence.agents.settings import AgentCfg, load_agents_settings
from precursorintelligence.agents.testing import ScriptedChatModel, Step, answer, call_tool, say
from precursorintelligence.agents.tools.evidence import Reason

V, GOLD = "2026-10", "20261111T000000Z-fixture"
DATA = DataVersion(v=V, evidence_manifest_sha="a" * 64)
RAISES = Reason(
    evidence_id=f"reason:015009:{V}:raises:1",
    v=V,
    gold_run_id=GOLD,
    ccn="015009",
    direction="raises",
    rank=1,
    feature="n_cit_GI_24m",
    label="Recent health citations",
    feature_value="7",
    contribution_logodds=0.42,
)


# ----------------------------------------------------------------------------- the models


def test_a_driver_is_built_from_riyas_reason_record():
    d = Driver.from_reason(RAISES)
    assert d == Driver(
        label="Recent health citations", direction="raises", evidence_id=RAISES.evidence_id
    )


def test_evidence_ids_of_keeps_order_and_drops_repeats():
    lowers = RAISES.model_copy(update={"evidence_id": f"reason:015009:{V}:lowers:1"})
    assert evidence_ids_of([RAISES, lowers, RAISES]) == [RAISES.evidence_id, lowers.evidence_id]


def test_the_output_models_refuse_unknown_fields_and_bad_values():
    with pytest.raises(ValidationError):
        Driver(label="x", direction="up", evidence_id="e")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Section(title="t", body="b", extra="no")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        AgentOutput(evidence_ids=[], verdict="approve")  # type: ignore[call-arg]


def test_every_output_carries_the_caveat_by_default():
    assert AgentOutput().caveat == CAVEAT
    assert "not a claims prediction" in CAVEAT.lower()


# ----------------------------------------------------------------------------- stamped by run_agent

METHODS = pytest.mark.parametrize("method", ["json_schema", "function_calling"])
HIDDEN = {"data_as_of", "caveat", "fallback"}


class Brief(AgentOutput):
    drivers: list[Driver]
    sections: list[Section]


class Q(BaseModel):
    ccn: str


@tool
async def get_reasons(ccn: str, deps: Annotated[RunDeps, InjectedToolArg]) -> list[Reason]:
    """The facility's drivers."""
    return [RAISES]


async def template(inp: Any, deps: Any) -> Brief:
    return Brief(
        drivers=[Driver.from_reason(RAISES)],
        sections=[
            Section(title="Why", body="Recent citations.", evidence_ids=[RAISES.evidence_id])
        ],
        evidence_ids=evidence_ids_of([RAISES]),
    )


def definition(*validators: Any) -> AgentDefinition[Q, Brief]:
    return AgentDefinition(
        name="demo",
        input_type=Q,
        output_type=Brief,
        fallback=template,
        tools=[get_reasons],
        validators=validators,
    )


def brief(driver_id: str, **extra: Any) -> Step:
    return answer(
        drivers=[
            {"label": "Recent health citations", "direction": "raises", "evidence_id": driver_id}
        ],
        sections=[{"title": "Why", "body": "Citations.", "evidence_ids": [RAISES.evidence_id]}],
        evidence_ids=[RAISES.evidence_id],
        **extra,
    )


def scripted(*answers: Step) -> ScriptedChatModel:
    """Calls get_reasons, then answers with each of ``answers`` (a failed answer fetches the reasons again)."""
    steps: list[Step] = [call_tool("get_reasons", ccn="015009")]
    for i, a in enumerate(answers):
        if i:
            steps.append(call_tool("get_reasons", ccn="015009"))
        steps.append(a)
    return ScriptedChatModel(steps={"*": steps})


def _runtime(tmp_path, model: ScriptedChatModel, method: str) -> Runtime:
    (tmp_path / "demo.yaml").write_text(
        "version: 1\nsystem: s\ninstructions: i\n", encoding="utf-8"
    )
    s = load_agents_settings()
    s.model.structured_output_method = method  # type: ignore[assignment]
    s.agents["demo"] = AgentCfg(max_tokens=500, temperature=0.0)
    return Runtime(
        settings=s,
        model=ModelPair(primary=model, primary_id=s.model.primary, fallback=None, fallback_id=None),
        cache=InMemoryAnswerCache(),
        counter=InMemoryDailyCounter(run_cap=10, spend_stop_usd=2.0),
        prompts_dir=tmp_path,
        today=lambda: date(2026, 11, 21),
    )


def _run(rt: Runtime, defn: AgentDefinition | None = None):
    return asyncio.run(
        run_agent(rt, defn or definition(), {"ccn": "015009"}, data=DATA, request_id="r")
    )


def _answer_schemas(model: ScriptedChatModel) -> list[dict[str, Any]]:
    """Every output schema the model was shown: the final_answer tool's, or the response_format's."""
    schemas = []
    for _, settings in model.calls:
        for t in settings.get("tools", []):
            if t["function"]["name"] == ANSWER_TOOL:
                schemas.append(t["function"]["parameters"])
        if fmt := (settings.get("extra_body") or {}).get("response_format"):
            schemas.append(fmt["json_schema"]["schema"])
    return schemas


@METHODS
def test_the_model_never_sees_data_as_of_caveat_or_fallback(tmp_path, method):
    model = scripted(brief(RAISES.evidence_id))
    if (
        method == "json_schema"
    ):  # a plain reply ends the tool phase, so the answer is asked for in the format
        model.steps["*"].insert(1, say("Here is my answer."))
    _run(_runtime(tmp_path, model, method))
    schemas = _answer_schemas(model)
    assert schemas
    for schema in schemas:
        props = set(schema["properties"])
        assert {"drivers", "sections", "evidence_ids"} <= props
        assert not props & HIDDEN
        assert not set(schema.get("required", [])) & HIDDEN


@METHODS
def test_a_model_answer_is_stamped_and_cached_stamped(tmp_path, method):
    # even a model that writes the hidden fields anyway can't change them
    model = scripted(
        brief(RAISES.evidence_id, data_as_of="1999-01", caveat="Will be cited.", fallback=True)
    )
    rt = _runtime(tmp_path, model, method)
    out = _run(rt)
    assert not out.fallback and out.cacheable
    assert (out.output.data_as_of, out.output.caveat, out.output.fallback) == (V, CAVEAT, False)
    again = _run(rt)
    assert again.cache_hit
    assert (again.output.data_as_of, again.output.caveat, again.output.fallback) == (
        V,
        CAVEAT,
        False,
    )


@METHODS
def test_the_validators_see_the_stamped_answer(tmp_path, method):
    seen: list[tuple[Any, ...]] = []

    def check(output: Brief, deps: RunDeps) -> None:
        seen.append((output.data_as_of, output.caveat, output.fallback))

    model = scripted(brief(RAISES.evidence_id, data_as_of="1999-01"))
    _run(_runtime(tmp_path, model, method), definition(check))
    assert seen == [(V, CAVEAT, False)]


@METHODS
def test_the_fallback_answer_is_stamped_too(tmp_path, method):
    made_up = f"reason:015009:{V}:raises:9"
    out = _run(_runtime(tmp_path, scripted(brief(made_up), brief(made_up)), method))
    assert out.fallback
    assert (out.output.data_as_of, out.output.caveat, out.output.fallback) == (V, CAVEAT, True)
    assert out.output.drivers == [Driver.from_reason(RAISES)]


def test_a_cache_hit_is_stamped_from_the_run(tmp_path):
    # an entry stored without the stamp (or with an old caveat) is served stamped, with no model call
    model = ScriptedChatModel(steps={"*": []})
    rt = _runtime(tmp_path, model, "json_schema")
    entry = asyncio.run(template(Q(ccn="015009"), None)).model_dump(mode="json")
    entry.update(data_as_of=None, caveat="An old caveat.", fallback=True)
    digest = build_cache_key(rt, definition(), Q(ccn="015009"), DATA).digest()
    asyncio.run(rt.cache.put(digest, entry))
    out = _run(rt)
    assert out.cache_hit and not model.calls
    assert (out.output.data_as_of, out.output.caveat, out.output.fallback) == (V, CAVEAT, False)


@METHODS
def test_a_made_up_id_inside_a_driver_is_rejected(tmp_path, method):
    made_up = f"reason:015009:{V}:raises:9"
    out = _run(_runtime(tmp_path, scripted(brief(made_up), brief(RAISES.evidence_id)), method))
    assert not out.fallback
    assert len(out.validation_failures) == 1 and made_up in out.validation_failures[0]
