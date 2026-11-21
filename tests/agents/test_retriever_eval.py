"""G5: the golden set file and the scorer."""

from __future__ import annotations

import pytest
from corpus_fakes import lookup_frame

from precursorintelligence.agents.corpus import (
    Corpus,
    corpus_version,
    passages_from_lookup,
    serialize_passages,
)
from precursorintelligence.agents.evidence_retriever import CAVEAT, EvidenceSet, Quote
from precursorintelligence.agents.retriever_eval import (
    GoldenCase,
    GoldenSet,
    load_golden,
    run_fallback,
    score_answers,
    score_retrieval,
)

F689 = "passage:cms_tag:F0689:0"


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    ps = passages_from_lookup(lookup_frame())
    return Corpus(version=corpus_version(serialize_passages(ps)), passages=tuple(ps))


def test_the_repo_golden_set():
    gs = load_golden()
    assert len(gs.cases) == 50
    assert sum(c.answerable for c in gs.cases) == 40
    kinds = {c.kind for c in gs.cases}
    assert {
        "plain",
        "tag_number",
        "claim",
        "as_of",
        "injection",
        "off_topic",
        "unknown_tag",
    } <= kinds
    assert all(c.as_of for c in gs.cases if c.kind == "as_of")


@pytest.mark.parametrize(
    "case",
    [
        dict(answerable=True, expected_tags=[], expected_passages=[]),
        dict(answerable=False, expected_tags=["F0689"], expected_passages=[], why_unanswerable="x"),
        dict(answerable=False, expected_tags=[], expected_passages=[]),
        dict(
            answerable=True, expected_tags=["F0689"], expected_passages=["passage:cms_tag:F0880:0"]
        ),
    ],
)
def test_inconsistent_cases_are_refused(case):
    with pytest.raises(ValueError):
        GoldenCase(id="er-001", question="q", kind="plain", **case)


def small_golden() -> GoldenSet:
    return GoldenSet(
        version=1,
        corpus_version="x",
        cases=[
            GoldenCase(
                id="er-001",
                question="accident hazards supervision",
                answerable=True,
                kind="regulation_wording",
                expected_tags=["F0689"],
                expected_passages=[F689],
            ),
            GoldenCase(
                id="er-002",
                question="weather in Boston tomorrow",
                answerable=False,
                kind="off_topic",
                expected_tags=[],
                expected_passages=[],
                why_unanswerable="off topic",
            ),
            GoldenCase(
                id="er-003",
                question="medications that restrain",
                answerable=True,
                kind="as_of",
                expected_tags=["F0605"],
                expected_passages=["passage:cms_tag:F0605:0"],
                as_of="2024-06",
            ),
        ],
    )


def _es(q, quotes=(), abstained=False, fallback=False) -> EvidenceSet:
    quotes = list(quotes)
    return EvidenceSet(
        question=q,
        abstained=abstained,
        abstain_reason="r" if abstained else None,
        quotes=quotes,
        claims=[],
        evidence=list(dict.fromkeys(x.passage_id for x in quotes)),
        corpus_version="x",
        data_as_of="2026-10",
        as_of=None,
        caveat=CAVEAT,
        prompt_version=1,
        fallback=fallback,
        validator_failures=[],
    )


def test_perfect_answers_meet_the_targets(corpus):
    gs = small_golden()
    answers = {
        "er-001": _es("q", [Quote(passage_id=F689, text="adequate supervision")]),
        "er-002": _es("q", abstained=True),
        "er-003": _es("q", [Quote(passage_id="passage:cms_tag:F0605:0", text="restrain them")]),
    }
    out = score_answers(gs, answers, corpus)
    assert (
        out["meets_targets"] and out["quote_fidelity"] == 1.0 and out["tag_hit_answerable"] == 1.0
    )


def test_an_edited_quote_or_a_wrong_abstention_misses_the_targets(corpus):
    gs = small_golden()
    answers = {
        "er-001": _es("q", [Quote(passage_id=F689, text="adequate supervisions")]),
        "er-002": _es("q", [Quote(passage_id=F689, text="adequate supervision")]),
        "er-003": _es("q", abstained=True),
    }
    out = score_answers(gs, answers, corpus)
    assert not out["meets_targets"]
    assert out["quote_fidelity"] < 1.0 and out["abstention_correct"] < 1.0
    assert set(out["misses"]) == {"er-002", "er-003"}


def test_a_missing_answer_is_an_error(corpus):
    with pytest.raises(ValueError, match="no answer"):
        score_answers(small_golden(), {}, corpus)


def test_retrieval_and_fallback_modes(corpus):
    gs = small_golden()
    r = score_retrieval(gs, corpus)
    assert r["hit_at_1"] == 1.0 and r["abstained_when_unanswerable"] == 1.0
    out = score_answers(gs, run_fallback(gs, corpus), corpus)
    assert out["fallback_rate"] == 1.0 and out["quote_fidelity"] == 1.0
    assert out["meets_targets"]
