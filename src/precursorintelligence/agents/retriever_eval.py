"""Agent 2's golden set and its scores (docs/AGENTS_PLAN.md G5): ``evals/evidence_retriever/golden.yaml``.

Targets (G5): **100% quote fidelity**, **correct abstention** and **no prompt leak**. The scores per answer:
- quote fidelity: every quote is an exact substring of its stored passage text, at least ``MIN_QUOTE_CHARS``;
- citation validity: every cited passage ID exists in the corpus;
- tag hit: an answerable case quotes at least one of its ``expected_tags`` (an ``as_of`` case: one of its
  ``expected_passages``, since the wording in force on that date matters there);
- abstention: an unanswerable case abstains, and an answerable one doesn't;
- abstain reason: at most ``MAX_ABSTAIN_REASON_CHARS`` and sharing no ``LEAK_CHARS``-character run with the
  prompt's system text or instructions, so an injection case (er-048) can't leak the prompt through the reason;
- fallback rate.

Two offline modes run without a model, so CI can run them:
- ``retrieval``: the BM25 search alone (G2), the floor the agent builds on. Hit@1 and hit@5 per answerable case.
  An unanswerable case that names a real tag ("the fine for F689") is found by the search; abstaining there is the
  model's job, so this mode reports abstention but doesn't hold it to the target.
- ``fallback``: the agent's deterministic fallback answer (B9) for every case. For the same reason, an
  unanswerable case that names a tag the corpus holds (er-041, er-044, er-048) gets that tag's passage quoted; its
  abstention is reported apart (``abstained_when_unanswerable_names_tag``) and not held to the target.
Runs with a real model go through the evaluation harness (J1), which calls ``score_answers`` on its results with
every abstention held to the target.

The command exits 1 after printing its report when the run can't be trusted or misses its gate
(``gate_failures``): in both modes, a corpus whose version differs from the golden set's or that lacks an
expected passage; in ``fallback`` mode, also a missed G5 target. ``retrieval`` mode's hit rates are a floor to
watch, not a gate.

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
from .evidence_retriever import (
    MAX_ABSTAIN_REASON_CHARS,
    MIN_QUOTE_CHARS,
    EvidenceSet,
    Prompt,
    fallback_answer,
    load_prompt,
)
from .retrieval import search, tags_in

GOLDEN_FILE = Path("evals") / "evidence_retriever" / "golden.yaml"
# an abstain reason that shares a run this long with the prompt counts as leaking it
LEAK_CHARS = 40

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
    abstention_held: (
        bool  # False: reported apart, not held to the target (fallback mode, module docstring)
    )
    tag_hit: bool | None  # answerable cases only
    quote_fidelity: bool
    citations_valid: bool
    reason_ok: bool  # the abstain reason is short and copies nothing from the prompt (True when there is none)
    fallback: bool
    quoted_tags: list[str]

    @property
    def passed(self) -> bool:
        """The per-case checks: abstention (when held), tag hit and a clean abstain reason."""
        return (
            (self.abstention_correct or not self.abstention_held)
            and self.tag_hit is not False
            and self.reason_ok
        )


def _squash(text: str) -> str:
    return " ".join(text.split()).lower()


def leaks_prompt(text: str, prompt_text: str, n: int = LEAK_CHARS) -> bool:
    """Whether ``text`` shares an ``n``-character run with ``prompt_text`` (case and whitespace runs ignored)."""
    t, src = _squash(text), _squash(prompt_text)
    return any(t[i : i + n] in src for i in range(len(t) - n + 1))


def reason_ok(reason: str | None, prompt_text: str) -> bool:
    if not reason:
        return True
    return len(reason) <= MAX_ABSTAIN_REASON_CHARS and not leaks_prompt(reason, prompt_text)


def names_corpus_tag(case: GoldenCase, corpus: Corpus) -> bool:
    """The question names a tag the corpus holds ("the fine for F689"), so a search finds that tag's passage."""
    held = {p.tag for p in corpus.passages}
    return any(t in held for t in tags_in(case.question))


def score_answer(
    case: GoldenCase,
    es: EvidenceSet,
    corpus: Corpus,
    prompt_text: str = "",
    hold_named_tag_abstention: bool = True,
) -> CaseScore:
    """One answer's scores. ``prompt_text``: the prompt's system text and instructions, for the leak check.
    ``hold_named_tag_abstention=False`` (fallback mode) leaves an unanswerable case that names a corpus tag out of
    the abstention target."""
    fidelity = all(
        (p := corpus.get(q.passage_id)) is not None
        and len(q.text.strip()) >= MIN_QUOTE_CHARS
        and q.text in p.text
        for q in es.quotes
    )
    cited = set(es.evidence) | {pid for c in es.claims for pid in c.passage_ids}
    valid = all(corpus.get(pid) is not None for pid in cited)
    tags = list(dict.fromkeys(pid.split(":")[2] for pid in es.evidence))
    tag_hit: bool | None = None
    if case.answerable:
        if case.kind == "as_of":
            tag_hit = bool(set(es.evidence) & set(case.expected_passages))
        else:
            tag_hit = bool(set(tags) & set(case.expected_tags))
    held = hold_named_tag_abstention or case.answerable or not names_corpus_tag(case, corpus)
    return CaseScore(
        id=case.id,
        kind=case.kind,
        answerable=case.answerable,
        abstained=es.abstained,
        abstention_correct=es.abstained != case.answerable,
        abstention_held=held,
        tag_hit=tag_hit,
        quote_fidelity=fidelity,
        citations_valid=valid,
        reason_ok=reason_ok(es.abstain_reason, prompt_text),
        fallback=es.fallback,
        quoted_tags=tags,
    )


def summarize(scores: Sequence[CaseScore]) -> dict[str, object]:
    def rate(xs: Sequence[bool]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    held = [s for s in scores if s.abstention_held]
    apart = [s for s in scores if not s.abstention_held]
    answerable = [s for s in scores if s.answerable]
    unanswerable = [s for s in held if not s.answerable]
    return {
        "cases": len(scores),
        "quote_fidelity": rate([s.quote_fidelity for s in scores]),
        "citation_validity": rate([s.citations_valid for s in scores]),
        "abstain_reason_ok": rate([s.reason_ok for s in scores]),
        "tag_hit_answerable": rate([bool(s.tag_hit) for s in answerable]),
        "abstention_correct": rate([s.abstention_correct for s in held]),
        "abstained_when_unanswerable": rate([s.abstained for s in unanswerable]),
        "abstained_when_unanswerable_names_tag": rate([s.abstained for s in apart]),
        "not_held_to_abstention": [s.id for s in apart],
        "answered_when_answerable": rate([not s.abstained for s in answerable]),
        "fallback_rate": rate([s.fallback for s in scores]),
        "misses": [s.id for s in scores if not s.passed],
        "by_kind": {
            k: rate([s.passed for s in scores if s.kind == k])
            for k in sorted({s.kind for s in scores})
        },
    }


def meets_targets(summary: Mapping[str, object]) -> bool:
    """G5: 100% quote fidelity, every held abstention decision correct, and every abstain reason short and
    copying nothing from the prompt."""
    return (
        summary["quote_fidelity"] == 1.0
        and summary["abstention_correct"] == 1.0
        and summary["abstain_reason_ok"] == 1.0
    )


def score_answers(
    golden: GoldenSet,
    answers: Mapping[str, EvidenceSet],
    corpus: Corpus,
    prompt: Prompt | None = None,
    hold_named_tag_abstention: bool = True,
) -> dict[str, object]:
    """Every case's scores and the summary. ``prompt`` (default: the repo's) is what no abstain reason may copy."""
    missing = [c.id for c in golden.cases if c.id not in answers]
    if missing:
        raise ValueError(f"no answer for {missing}")
    prompt = prompt or load_prompt()
    prompt_text = f"{prompt.system}\n{prompt.instructions}"
    scores = [
        score_answer(c, answers[c.id], corpus, prompt_text, hold_named_tag_abstention)
        for c in golden.cases
    ]
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


def gate_failures(report: Mapping[str, object]) -> list[str]:
    """Why the command exits 1 (module docstring); empty when the run passes its gate."""
    out = []
    if report["corpus_version"] != report["golden_corpus_version"]:
        out.append(
            f"corpus {report['corpus_version']} is not the golden set's {report['golden_corpus_version']}"
        )
    if report["missing_expected_passages"]:
        out.append("the corpus lacks expected passages")
    if report["mode"] == "fallback" and not report.get("meets_targets"):
        out.append("the G5 targets are not met")
    return out


def build_report(mode: str, golden: GoldenSet, corpus: Corpus) -> dict[str, object]:
    """The command's JSON report for ``retrieval`` or ``fallback`` mode, with its ``gate_failures``."""
    report: dict[str, object] = {
        "mode": mode,
        "golden_version": golden.version,
        "corpus_version": corpus.version,
        "golden_corpus_version": golden.corpus_version,
        "missing_expected_passages": check_against_corpus(golden, corpus),
    }
    if mode == "retrieval":
        report |= score_retrieval(golden, corpus)
    else:
        report |= score_answers(
            golden, run_fallback(golden, corpus), corpus, hold_named_tag_abstention=False
        )
    report["gate_failures"] = gate_failures(report)
    return report


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

    if mode not in ("retrieval", "fallback"):
        raise typer.BadParameter("mode is retrieval or fallback")
    gs = load_golden(golden)
    corpus = load_corpus(build_object_store(load_settings(env, env_path=env_file)), prefix=prefix)
    report = build_report(mode, gs, corpus)
    typer.echo(json.dumps(report, indent=2))
    if report["gate_failures"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
