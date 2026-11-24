"""The shared output models (docs/AGENTS_PLAN.md A5) and how run_agent stamps them. No paid model, no network."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Annotated, Any

import pytest
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, ValidationError
from pydantic.json_schema import SkipJsonSchema

from precursorintelligence.agents.cache import InMemoryAnswerCache
from precursorintelligence.agents.counter import InMemoryDailyCounter
from precursorintelligence.agents.ledger import prose_evidence_ids
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


# ----------------------------------------------------------------------------- per-agent caveat and data month

RETRIEVER_CAVEAT = "Quotes are CMS's short tag descriptions, not the full regulation text."
CORPUS_MONTH = "2025-10"


class EvidenceSetLike(AgentOutput):
    """Shaped like Agent 2's output once ported: its own caveat and data month."""

    caveat: SkipJsonSchema[str] = RETRIEVER_CAVEAT
    summary: str


async def retriever_template(inp: Any, deps: Any) -> EvidenceSetLike:
    return EvidenceSetLike(summary="No passage found.", evidence_ids=[])


def retriever_definition() -> AgentDefinition[Q, EvidenceSetLike]:
    return AgentDefinition(
        name="demo",
        input_type=Q,
        output_type=EvidenceSetLike,
        fallback=retriever_template,
        tools=[get_reasons],
        data_as_of=lambda data, tools: CORPUS_MONTH,
    )


@METHODS
def test_an_agent_keeps_its_own_caveat_and_data_month(tmp_path, method):
    model = scripted(
        answer(summary="Recent citations.", evidence_ids=[RAISES.evidence_id], caveat="x")
    )
    rt = _runtime(tmp_path, model, method)
    out = _run(rt, retriever_definition())
    assert not out.fallback
    assert (out.output.data_as_of, out.output.caveat) == (CORPUS_MONTH, RETRIEVER_CAVEAT)
    again = _run(rt, retriever_definition())
    assert again.cache_hit
    assert (again.output.data_as_of, again.output.caveat) == (CORPUS_MONTH, RETRIEVER_CAVEAT)


def test_an_agents_fallback_keeps_its_own_caveat_and_data_month(tmp_path):
    made_up = f"reason:015009:{V}:raises:9"
    bad = answer(summary="s", evidence_ids=[made_up])
    out = _run(_runtime(tmp_path, scripted(bad, bad), "json_schema"), retriever_definition())
    assert out.fallback
    assert (out.output.data_as_of, out.output.caveat, out.output.fallback) == (
        CORPUS_MONTH,
        RETRIEVER_CAVEAT,
        True,
    )


# ----------------------------------------------------------------------------- the month check

OLD = RAISES.model_copy(update={"evidence_id": "reason:015009:2026-09:raises:1", "v": "2026-09"})
PASSAGE_ID = "passage:cms_tag:F0689:2021-01"


@tool
async def get_old_and_new_reasons(
    ccn: str, deps: Annotated[RunDeps, InjectedToolArg]
) -> list[Reason]:
    """The facility's drivers, one of them from an older month."""
    return [RAISES, OLD]


@tool
async def find_regulations(ccn: str, deps: Annotated[RunDeps, InjectedToolArg]) -> list[Any]:
    """The facility's drivers and what Agent 2 found: a passage whose wording starts in 2021-01."""
    found = EvidenceSetLike(
        summary="F0689 accidents.", evidence_ids=[PASSAGE_ID], data_as_of=CORPUS_MONTH
    )
    return [RAISES, found]


def _with_tool(t: Any, *validators: Any) -> AgentDefinition[Q, Brief]:
    return AgentDefinition(
        name="demo",
        input_type=Q,
        output_type=Brief,
        fallback=template,
        tools=[t],
        validators=validators,
    )


def _scripted_with(tool_name: str, *answers: Step) -> ScriptedChatModel:
    steps: list[Step] = []
    for a in answers:
        steps += [call_tool(tool_name, ccn="015009"), a]
    return ScriptedChatModel(steps={"*": steps})


@METHODS
def test_a_record_from_another_month_is_rejected(tmp_path, method):
    model = _scripted_with(
        "get_old_and_new_reasons", brief(OLD.evidence_id), brief(RAISES.evidence_id)
    )
    out = _run(_runtime(tmp_path, model, method), _with_tool(get_old_and_new_reasons))
    assert not out.fallback
    assert len(out.validation_failures) == 1
    assert (
        OLD.evidence_id in out.validation_failures[0]
        and "another month" in out.validation_failures[0]
    )


@METHODS
def test_agent_1_can_cite_a_passage_from_2021_01(tmp_path, method):
    # a passage is dated by the vintage its wording starts, not by the score month, so it isn't month-checked
    cites = answer(
        drivers=[
            {
                "label": "Recent health citations",
                "direction": "raises",
                "evidence_id": RAISES.evidence_id,
            }
        ],
        sections=[
            {"title": "Why", "body": f"Accidents, per {PASSAGE_ID}.", "evidence_ids": [PASSAGE_ID]},
        ],
        evidence_ids=[RAISES.evidence_id, PASSAGE_ID],
    )
    out = _run(
        _runtime(tmp_path, _scripted_with("find_regulations", cites), method),
        _with_tool(find_regulations),
    )
    assert not out.fallback and not out.validation_failures
    assert out.output.data_as_of == V
    assert PASSAGE_ID in out.output.evidence_ids


# ----------------------------------------------------------------------------- IDs written in the text


def test_prose_evidence_ids_finds_ids_in_text_only():
    text = (
        f"Driven by {RAISES.evidence_id}. Risk score: 0.31, see {PASSAGE_ID}, and score:015009:{V}."
    )
    assert list(prose_evidence_ids(text)) == [RAISES.evidence_id, PASSAGE_ID, f"score:015009:{V}"]
    section = Section(
        title="t", body="no IDs, a score: 0.31", evidence_ids=["score:015009:2026-10"]
    )
    assert (
        list(prose_evidence_ids(section)) == []
    )  # the evidence_ids list is validator 1's ordinary check


@METHODS
def test_a_made_up_id_written_in_the_text_is_rejected(tmp_path, method):
    made_up = f"reason:015393:{V}:raises:1"
    in_text = answer(
        drivers=[
            {
                "label": "Recent health citations",
                "direction": "raises",
                "evidence_id": RAISES.evidence_id,
            }
        ],
        sections=[
            {
                "title": "Why",
                "body": f"Citations, per {made_up}.",
                "evidence_ids": [RAISES.evidence_id],
            }
        ],
        evidence_ids=[RAISES.evidence_id],
    )
    out = _run(_runtime(tmp_path, scripted(in_text, brief(RAISES.evidence_id)), method))
    assert not out.fallback
    assert len(out.validation_failures) == 1
    assert (
        made_up in out.validation_failures[0]
        and "written in the text" in out.validation_failures[0]
    )


@METHODS
def test_a_real_id_written_in_the_text_passes(tmp_path, method):
    in_text = answer(
        drivers=[
            {
                "label": "Recent health citations",
                "direction": "raises",
                "evidence_id": RAISES.evidence_id,
            }
        ],
        sections=[
            {"title": "Why", "body": f"Citations, per {RAISES.evidence_id}.", "evidence_ids": []}
        ],
        evidence_ids=[RAISES.evidence_id],
    )
    out = _run(_runtime(tmp_path, scripted(in_text), method))
    assert not out.fallback and not out.validation_failures
