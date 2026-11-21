"""Agent 2's golden set and its scores (docs/AGENTS_PLAN.md G5): ``evals/evidence_retriever/golden.yaml``.

Targets (G5): **100% quote fidelity** and **correct abstention**. The scores per answer:
- quote fidelity: every quote is an exact substring of its stored passage text;
- citation validity: every cited passage ID exists in the corpus;
- tag hit: an answerable case quotes at least one of its ``expected_tags``;
- abstention: an unanswerable case abstains, and an answerable one doesn't;
- fallback rate.

Two offline modes run without a model, so CI can run them:
- ``retrieval``: the BM25 search alone (G2), the floor the agent builds on. Hit@1 and hit@5 per answerable case.
  An unanswerable case that names a real tag ("the fine for F689") is found by the search; abstaining there is the
  model's job, so this mode reports abstention but doesn't hold it to the target.
- ``fallback``: the agent's deterministic fallback answer (B9) for every case.
Runs with a real model go through the evaluation harness (J1), which calls ``score_answers`` on its results.

    python -m precursorintelligence.agents.retriever_eval --mode retrieval --env-file data/local_env.yaml
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import typer
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..common.config import project_root
from .corpus import DEFAULT_PREFIX, Corpus, load_corpus
from .evidence_retriever import EvidenceSet, fallback_answer
from .retrieval import search

GOLDEN_FILE = Path("evals") / "evidence_retriever" / "golden.yaml"

Kind = Literal[
    "plain",
    "regulation_wording",
    "tag_number",
    "claim",
    "as_of",
    "missing_fact",
    "advice",
    "off_topic",
    "injection",
    "unknown_tag",
    "uncovered_topic",
]


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^er-\d{3}$")
    question: str = Field(min_length=1, max_length=500)
    answerable: bool
    kind: Kind
    expected_tags: list[str]
    expected_passages: list[str]
    as_of: str | None = None
    why_unanswerable: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> GoldenCase:
        if self.answerable and not self.expected_tags:
            raise ValueError(f"{self.id}: an answerable case needs expected_tags")
        if not self.answerable and (self.expected_tags or self.expected_passages):
            raise ValueError(f"{self.id}: an unanswerable case expects no tags or passages")
        if not self.answerable and not self.why_unanswerable:
            raise ValueError(f"{self.id}: an unanswerable case needs why_unanswerable")
        for pid in self.expected_passages:
            if pid.split(":")[2] not in self.expected_tags:
                raise ValueError(f"{self.id}: {pid} is not one of its expected_tags")
        return self


class GoldenSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    corpus_version: str
    cases: list[GoldenCase]

    @model_validator(mode="after")
    def _unique(self) -> GoldenSet:
        ids = [c.id for c in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate case ids")
        return self


def load_golden(path: Path | None = None) -> GoldenSet:
    path = path or project_root() / GOLDEN_FILE
    return GoldenSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def check_against_corpus(golden: GoldenSet, corpus: Corpus) -> list[str]:
    """Expected passages the corpus doesn't hold (after a corpus rebuild, the golden set must be rechecked)."""
    return [
        f"{c.id}: {pid}"
        for c in golden.cases
        for pid in c.expected_passages
        if corpus.get(pid) is None
    ]


# ---------------------------------------------------------------- scoring


class CaseScore(BaseModel):
    id: str
    kind: str
    answerable: bool
    abstained: bool
    abstention_correct: bool
    tag_hit: bool | None  # answerable cases only
    quote_fidelity: bool
    citations_valid: bool
    fallback: bool
    quoted_tags: list[str]


def score_answer(case: GoldenCase, es: EvidenceSet, corpus: Corpus) -> CaseScore:
    fidelity = all(
        (p := corpus.get(q.passage_id)) is not None and q.text in p.text for q in es.quotes
    )
    cited = set(es.evidence) | {pid for c in es.claims for pid in c.passage_ids}
    valid = all(corpus.get(pid) is not None for pid in cited)
    tags = list(dict.fromkeys(pid.split(":")[2] for pid in es.evidence))
    return CaseScore(
        id=case.id,
        kind=case.kind,
        answerable=case.answerable,
        abstained=es.abstained,
        abstention_correct=es.abstained != case.answerable,
        tag_hit=bool(set(tags) & set(case.expected_tags)) if case.answerable else None,
        quote_fidelity=fidelity,
        citations_valid=valid,
        fallback=es.fallback,
        quoted_tags=tags,
    )


def summarize(scores: Sequence[CaseScore]) -> dict[str, object]:
    def rate(xs: Sequence[bool]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    answerable = [s for s in scores if s.answerable]
    unanswerable = [s for s in scores if not s.answerable]
    return {
        "cases": len(scores),
        "quote_fidelity": rate([s.quote_fidelity for s in scores]),
        "citation_validity": rate([s.citations_valid for s in scores]),
        "tag_hit_answerable": rate([bool(s.tag_hit) for s in answerable]),
        "abstention_correct": rate([s.abstention_correct for s in scores]),
        "abstained_when_unanswerable": rate([s.abstained for s in unanswerable]),
        "answered_when_answerable": rate([not s.abstained for s in answerable]),
        "fallback_rate": rate([s.fallback for s in scores]),
        "misses": [s.id for s in scores if not s.abstention_correct or s.tag_hit is False],
        "by_kind": {
            k: rate(
                [s.abstention_correct and s.tag_hit is not False for s in scores if s.kind == k]
            )
            for k in sorted({s.kind for s in scores})
        },
    }


def meets_targets(summary: Mapping[str, object]) -> bool:
    """G5: 100% quote fidelity and every abstention decision correct."""
    return summary["quote_fidelity"] == 1.0 and summary["abstention_correct"] == 1.0


def score_answers(
    golden: GoldenSet, answers: Mapping[str, EvidenceSet], corpus: Corpus
) -> dict[str, object]:
    missing = [c.id for c in golden.cases if c.id not in answers]
    if missing:
        raise ValueError(f"no answer for {missing}")
    scores = [score_answer(c, answers[c.id], corpus) for c in golden.cases]
    out = summarize(scores)
    out["meets_targets"] = meets_targets(out)
    return out


def score_retrieval(golden: GoldenSet, corpus: Corpus, k: int = 5) -> dict[str, object]:
    """The BM25 search alone: hit@1 and hit@k on answerable cases, and how often it abstains on the others."""
    hit1: list[bool] = []
    hitk: list[bool] = []
    abstained_unanswerable: list[bool] = []
    misses: list[str] = []
    kinds: Counter[str] = Counter()
    for c in golden.cases:
        res = search(corpus, c.question, k=k, as_of=c.as_of)
        tags = [h.passage.tag for h in res.hits]
        if c.answerable:
            ok1 = bool(tags) and tags[0] in c.expected_tags
            okk = bool(set(tags) & set(c.expected_tags))
            if c.kind == "as_of":
                ids = [h.passage.passage_id for h in res.hits]
                ok1 = bool(ids) and ids[0] in c.expected_passages
                okk = bool(set(ids) & set(c.expected_passages))
            hit1.append(ok1)
            hitk.append(okk)
            if not okk:
                misses.append(c.id)
                kinds[c.kind] += 1
        else:
            abstained_unanswerable.append(res.abstained)

    def rate(xs: list[bool]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "cases": len(golden.cases),
        "hit_at_1": rate(hit1),
        f"hit_at_{k}": rate(hitk),
        "abstained_when_unanswerable": rate(abstained_unanswerable),
        "misses": misses,
        "misses_by_kind": dict(kinds),
    }


def run_fallback(golden: GoldenSet, corpus: Corpus) -> dict[str, EvidenceSet]:
    """Every case answered by the deterministic fallback, as ``EvidenceSet``s the scorer reads."""
    from .evidence_retriever import CAVEAT, _data_as_of, _evidence_ids

    out = {}
    for c in golden.cases:
        a = fallback_answer(c.question, corpus, c.as_of)
        out[c.id] = EvidenceSet(
            question=c.question,
            abstained=a.abstained,
            abstain_reason=a.abstain_reason,
            quotes=a.quotes,
            claims=a.claims,
            evidence=_evidence_ids(a),
            corpus_version=corpus.version,
            data_as_of=_data_as_of(corpus),
            as_of=c.as_of,
            caveat=CAVEAT,
            prompt_version=0,
            fallback=True,
            validator_failures=[],
        )
    return out


# ---------------------------------------------------------------- command line

app = typer.Typer(add_completion=False)


@app.command()
def main(
    mode: str = typer.Option(
        "retrieval", help="retrieval (BM25 alone) or fallback (the agent's fallback)"
    ),
    env: str = typer.Option("dev", help="environment config: configs/<env>.yaml"),
    env_file: Path | None = typer.Option(None, help="explicit env yaml (overrides --env)"),
    prefix: str = typer.Option(DEFAULT_PREFIX, help="the corpus prefix in the object store"),
    golden: Path | None = typer.Option(
        None, help="golden set (default: evals/evidence_retriever/golden.yaml)"
    ),
) -> None:
    """Score Agent 2's golden set offline; prints one JSON report."""
    from ..common.config import load_settings
    from ..common.io.registry import build_object_store

    gs = load_golden(golden)
    corpus = load_corpus(build_object_store(load_settings(env, env_path=env_file)), prefix=prefix)
    report: dict[str, object] = {
        "mode": mode,
        "golden_version": gs.version,
        "corpus_version": corpus.version,
        "golden_corpus_version": gs.corpus_version,
        "missing_expected_passages": check_against_corpus(gs, corpus),
    }
    if mode == "retrieval":
        report |= score_retrieval(gs, corpus)
    elif mode == "fallback":
        report |= score_answers(gs, run_fallback(gs, corpus), corpus)
    else:
        raise typer.BadParameter("mode is retrieval or fallback")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
