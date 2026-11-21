"""G3/G4: Agent 2 on scripted models (no paid model): validated answers, retry, fallback, abstention."""

from __future__ import annotations

from pathlib import Path

import pytest
from corpus_fakes import lookup_frame
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from precursorintelligence.agents.corpus import (
    Corpus,
    corpus_version,
    passages_from_lookup,
    serialize_passages,
)
from precursorintelligence.agents.evidence_retriever import (
    CAVEAT,
    Claim,
    EvidenceAnswer,
    PromptError,
    Quote,
    check_question,
    fallback_answer,
    load_prompt,
    run_evidence_retriever,
    validate_answer,
)

F689 = "passage:cms_tag:F0689:2021-01"
F689_TEXT = (
    "Ensure that a nursing home area is free from accident hazards and provides adequate supervision to "
    "prevent accidents."
)


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    ps = passages_from_lookup(lookup_frame())
    return Corpus(version=corpus_version(serialize_passages(ps)), passages=tuple(ps))


def _search_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [
        p
        for m in messages
        for p in getattr(m, "parts", [])
        if isinstance(p, ToolReturnPart) and p.tool_name == "search_passages"
    ]


def _retry_prompts(messages: list[ModelMessage]) -> int:
    return sum(
        1
        for m in messages
        for p in getattr(m, "parts", [])
        if getattr(p, "part_kind", "") == "retry-prompt"
    )


def scripted(
    answers: list[dict], query: str = "accident hazards supervision F689"
) -> FunctionModel:
    """A model that searches once, then gives ``answers`` in turn (one per attempt)."""

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not _search_returns(messages):
            return ModelResponse(parts=[ToolCallPart("search_passages", {"query": query})])
        attempt = min(_retry_prompts(messages), len(answers) - 1)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answers[attempt])])

    return FunctionModel(fn)


GOOD = {
    "abstained": False,
    "quotes": [
        {"passage_id": F689, "text": "free from accident hazards and provides adequate supervision"}
    ],
    "claims": [
        {
            "claim": "Homes must prevent accident hazards and supervise residents.",
            "passage_ids": [F689],
        }
    ],
}
EDITED = {
    "abstained": False,
    "quotes": [{"passage_id": F689, "text": "free from all accident hazards"}],
    "claims": [],
}
MADE_UP = {
    "abstained": False,
    "quotes": [
        {"passage_id": "passage:cms_tag:F0880:2021-01", "text": "infection prevention and control"}
    ],
    "claims": [],
}


def test_a_valid_answer_is_accepted(corpus):
    es = run_evidence_retriever(
        "Is there a rule about residents falling?", corpus, scripted([GOOD])
    )
    assert not es.fallback and not es.abstained
    assert es.evidence == [F689]
    assert es.quotes[0].text in F689_TEXT
    assert es.corpus_version == corpus.version
    assert es.data_as_of == "2026-10"
    assert es.caveat == CAVEAT and es.prompt_version >= 1
    assert es.validator_failures == []


def test_an_edited_quote_gets_one_retry_then_passes(corpus):
    es = run_evidence_retriever("rule about falls?", corpus, scripted([EDITED, GOOD]))
    assert not es.fallback
    assert es.evidence == [F689]
    assert any("exact substring" in f for f in es.validator_failures)


def test_an_id_no_tool_returned_is_rejected(corpus):
    """F0880 is in the corpus, but no tool returned it in this run (B3)."""
    es = run_evidence_retriever("rule about falls?", corpus, scripted([MADE_UP, GOOD]))
    assert not es.fallback and es.evidence == [F689]
    assert any("not returned by a tool" in f for f in es.validator_failures)


def test_two_failures_give_the_fallback(corpus):
    es = run_evidence_retriever("accident hazards supervision", corpus, scripted([EDITED, EDITED]))
    assert es.fallback
    assert es.evidence and es.evidence[0] == F689
    assert all(q.text == corpus.get(q.passage_id).text for q in es.quotes)  # quoted whole


def test_the_fallback_abstains_when_the_search_does(corpus):
    es = run_evidence_retriever("weather in Boston tomorrow", corpus, scripted([EDITED, EDITED]))
    assert es.fallback and es.abstained and es.quotes == [] and es.abstain_reason


def test_an_abstention_is_accepted(corpus):
    answer = {
        "abstained": True,
        "abstain_reason": "Fine amounts are facility data.",
        "quotes": [],
        "claims": [],
    }
    es = run_evidence_retriever("What was the fine for F689?", corpus, scripted([answer]))
    assert es.abstained and not es.fallback and es.evidence == []


def test_test_model_runs_end_to_end_and_returns_a_valid_set(corpus):
    es = run_evidence_retriever("accident hazards supervision", corpus, TestModel())
    for q in es.quotes:
        assert q.text in corpus.get(q.passage_id).text


def test_the_tools_put_ids_in_the_ledger_and_get_passage_works(corpus):
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        got = [
            p
            for m in messages
            for p in getattr(m, "parts", [])
            if isinstance(p, ToolReturnPart) and p.tool_name == "get_passage"
        ]
        if not got:
            return ModelResponse(parts=[ToolCallPart("get_passage", {"passage_id": F689})])
        quote = {"passage_id": F689, "text": got[0].content.text}
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"abstained": False, "quotes": [quote]})]
        )

    es = run_evidence_retriever("What does F689 say?", corpus, FunctionModel(fn))
    assert not es.fallback and es.quotes[0].text == F689_TEXT


# ---------------------------------------------------------------- the validator on its own


def test_validate_answer_rules(corpus):
    ledger = {F689}
    ok = EvidenceAnswer(
        abstained=False,
        quotes=[Quote(passage_id=F689, text="adequate supervision")],
        claims=[Claim(claim="Supervise residents.", passage_ids=[F689])],
    )
    assert validate_answer(ok, corpus, ledger) == []
    short = EvidenceAnswer(abstained=False, quotes=[Quote(passage_id=F689, text="Ensure")])
    assert "shorter" in validate_answer(short, corpus, ledger)[0]
    none = EvidenceAnswer(abstained=False)
    assert "at least one quote" in validate_answer(none, corpus, ledger)[0]
    unquoted_claim = EvidenceAnswer(
        abstained=False,
        quotes=[Quote(passage_id=F689, text="adequate supervision")],
        claims=[Claim(claim="x", passage_ids=["passage:cms_tag:F0880:2021-01"])],
    )
    assert "not a valid quote" in validate_answer(unquoted_claim, corpus, ledger)[0]
    no_reason = EvidenceAnswer(abstained=True)
    assert "abstain_reason" in validate_answer(no_reason, corpus, ledger)[0]
    abstain_with_quote = EvidenceAnswer(
        abstained=True,
        abstain_reason="r",
        quotes=[Quote(passage_id=F689, text="adequate supervision")],
    )
    assert "no quotes" in validate_answer(abstain_with_quote, corpus, ledger)[0]


def test_fallback_answer_quotes_whole_passages(corpus):
    a = fallback_answer("pressure ulcer care", corpus)
    assert not a.abstained and a.quotes[0].passage_id == "passage:cms_tag:F0686:2021-01"
    assert a.quotes[0].text == corpus.get(a.quotes[0].passage_id).text


@pytest.mark.parametrize("q", ["", "   ", "x" * 501])
def test_bad_question_is_refused_before_any_model_call(corpus, q):
    with pytest.raises(ValueError):
        check_question(q)
    with pytest.raises(ValueError):
        run_evidence_retriever(q, corpus, TestModel())


# ---------------------------------------------------------------- the prompt


def test_the_repo_prompt_loads_with_a_version():
    p = load_prompt()
    assert p.version >= 1 and "abstain" in p.system.lower() and "search_passages" in p.instructions


@pytest.mark.parametrize(
    "text",
    [
        "system: s\ninstructions: i\n",
        "version: '1'\nsystem: s\ninstructions: i\n",
        "version: 0\nsystem: s\ninstructions: i\n",
        "version: 1\nsystem: ''\ninstructions: i\n",
        "- a list\n",
    ],
)
def test_a_prompt_without_a_valid_version_or_text_is_refused(tmp_path: Path, text):
    path = tmp_path / "p.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(PromptError):
        load_prompt(path)
