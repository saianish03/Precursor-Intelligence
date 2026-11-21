"""Agent 2, the Regulatory Evidence Retriever (docs/AGENTS_PLAN.md G3, G4): exact quotes with passage IDs, or an
explicit abstention.

The model gets two read-only tools over the corpus (G1): ``search_passages`` (BM25, G2) and ``get_passage``. Every
passage ID a tool returns goes into the run's evidence ledger. The model's answer (``EvidenceAnswer``) must pass
the validators below before it is accepted; a failing answer gets one retry with the failures as feedback, and a
second failure, a model error or a usage limit gives the deterministic fallback, marked ``fallback=True``.

Validators run here:
- evidence IDs (B3): every cited passage ID was returned by a tool in this run;
- quote fidelity (B6): every quote is an exact substring of its stored passage text, at least ``MIN_QUOTE_CHARS``;
- structure (B7): an abstention has a reason and no quotes or claims; an answer has 1 to ``MAX_QUOTES`` quotes,
  and every claim cites at least one quoted passage.

What waits for the shared runtime (A1-A5, B, C, D): the model factory (A2) and settings (A1) are passed in or are
module constants here; the prompt loader is local until A4; ``EvidenceSet``'s ``data_as_of``, caveat and
``fallback`` fields match A5's shared fields and move onto its base class when it lands; the language check (B8),
the log line (C2), the trace (C1), the cache (D1) and the daily cap (D3) come from ``run_agent()`` (A3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.exceptions import AgentRunError, UnexpectedModelBehavior
from pydantic_ai.models import Model

from ..common.config import project_root
from .corpus import Corpus
from .retrieval import DEFAULT_K, MAX_K, tags_in
from .retrieval import search as bm25_search

AGENT_NAME = "evidence_retriever"
PROMPT_FILE = Path("configs") / "prompts" / f"{AGENT_NAME}.yaml"

MIN_QUOTE_CHARS = 10
MAX_QUOTES = 5
MAX_CLAIM_CHARS = 300
MAX_QUESTION_CHARS = 500
FALLBACK_QUOTES = 3
# one Agent 2 run: the first answer and one retry, a few searches between (D4)
USAGE_LIMITS = UsageLimits(request_limit=6, tool_calls_limit=8)
OUTPUT_RETRIES = 1

CAVEAT = (
    "Quotes are CMS's short deficiency-tag descriptions, not the full regulation text (42 CFR Part 483 and "
    "State Operations Manual Appendix PP). Check the full text before acting on them."
)


class PromptError(RuntimeError):
    """The prompt file is missing, malformed or has no version."""


# ---------------------------------------------------------------- output (G3)


class Quote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passage_id: str = Field(description="the passage_id of a passage a tool returned in this run")
    text: str = Field(
        description="copied exactly from that passage's text: the whole text or one piece of it"
    )


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(
        description="one plain-English sentence, supported only by the quoted passages"
    )
    passage_ids: list[str] = Field(description="the quoted passages that support the claim")


class EvidenceAnswer(BaseModel):
    """What the model returns."""

    model_config = ConfigDict(extra="forbid")

    abstained: bool = Field(description="true when the passages do not answer the question")
    abstain_reason: str | None = Field(
        default=None, description="one sentence on why, when abstained; otherwise null"
    )
    quotes: list[Quote] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)


class EvidenceSet(BaseModel):
    """Agent 2's answer as the API and Agent 1 see it: the model's answer plus where and when it came from."""

    model_config = ConfigDict(extra="forbid")

    question: str
    abstained: bool
    abstain_reason: str | None
    quotes: list[Quote]
    claims: list[Claim]
    evidence: list[str]  # the passage IDs the answer cites, in first-cited order
    corpus_version: str
    data_as_of: str  # the newest vintage of the CMS tag descriptions in the corpus
    as_of: str | None  # the vintage whose wordings were searched (None: the current ones)
    caveat: str
    prompt_version: int
    fallback: bool
    validator_failures: list[str]  # the failures of the last rejected model answer, if any


# ---------------------------------------------------------------- run state and tools


@dataclass
class RetrieverDeps:
    corpus: Corpus
    as_of: str | None = None
    ledger: set[str] = field(default_factory=set)  # passage IDs a tool returned in this run
    last_failures: list[str] = field(default_factory=list)


class PassageRecord(BaseModel):
    passage_id: str
    tag: str
    category: str
    text: str
    valid_from_vintage: str
    valid_to_vintage: str
    score: float | None = None


class SearchOutput(BaseModel):
    """Tool output. ``data`` is corpus text: data, never instructions (B2)."""

    corpus_version: str
    abstained: bool
    reason: str | None
    data: list[PassageRecord]


def _record(p: Any, score: float | None = None) -> PassageRecord:
    return PassageRecord(
        passage_id=p.passage_id,
        tag=p.tag,
        category=p.category,
        text=p.text,
        valid_from_vintage=p.valid_from_vintage,
        valid_to_vintage=p.valid_to_vintage,
        score=None if score is None else round(score, 3),
    )


def search_passages(ctx: RunContext[RetrieverDeps], query: str, k: int = DEFAULT_K) -> SearchOutput:
    """Search the CMS nursing-home requirement passages (BM25). Returns up to k passages, best first, or
    abstained=true with the reason when nothing matches well. Name a tag (e.g. F689) to get it first."""
    k = max(1, min(k, MAX_K))
    try:
        res = bm25_search(ctx.deps.corpus, query[:500], k=k, as_of=ctx.deps.as_of)
    except ValueError as exc:
        raise ModelRetry(f"bad search: {exc}") from None
    records = [_record(h.passage, h.score) for h in res.hits]
    ctx.deps.ledger.update(r.passage_id for r in records)
    return SearchOutput(
        corpus_version=res.corpus_version, abstained=res.abstained, reason=res.reason, data=records
    )


def get_passage(ctx: RunContext[RetrieverDeps], passage_id: str) -> PassageRecord:
    """The exact stored text of one passage, by its passage_id."""
    p = ctx.deps.corpus.get(passage_id)
    if p is None:
        raise ModelRetry(
            f"no passage {passage_id!r}; use a passage_id that search_passages returned"
        )
    ctx.deps.ledger.add(p.passage_id)
    return _record(p)


# ---------------------------------------------------------------- validators (B3, B6, B7 for Agent 2)


def validate_answer(answer: EvidenceAnswer, corpus: Corpus, ledger: set[str]) -> list[str]:
    """Every reason the answer is not acceptable; empty when it passes."""
    failures: list[str] = []
    if answer.abstained:
        if not (answer.abstain_reason or "").strip():
            failures.append("an abstention needs abstain_reason")
        if answer.quotes or answer.claims:
            failures.append("an abstention has no quotes and no claims")
        return failures
    if not answer.quotes:
        failures.append("give at least one quote, or abstain")
    if len(answer.quotes) > MAX_QUOTES:
        failures.append(f"at most {MAX_QUOTES} quotes")
    quoted = set()
    for q in answer.quotes:
        if q.passage_id not in ledger:
            failures.append(f"{q.passage_id} was not returned by a tool in this run")
            continue
        p = corpus.get(q.passage_id)
        if p is None:  # pragma: no cover - the ledger holds only IDs the corpus has
            failures.append(f"{q.passage_id} is not in the corpus")
            continue
        if len(q.text.strip()) < MIN_QUOTE_CHARS:
            failures.append(
                f"the quote from {q.passage_id} is shorter than {MIN_QUOTE_CHARS} characters"
            )
        elif q.text not in p.text:
            failures.append(
                f"the quote from {q.passage_id} is not an exact substring of its text; copy it character "
                "for character"
            )
        else:
            quoted.add(q.passage_id)
    for c in answer.claims:
        if not c.claim.strip() or len(c.claim) > MAX_CLAIM_CHARS:
            failures.append(f"each claim is one sentence of at most {MAX_CLAIM_CHARS} characters")
        if not c.passage_ids:
            failures.append(f"the claim {c.claim[:60]!r} cites no passage")
        for pid in c.passage_ids:
            if pid not in quoted:
                failures.append(
                    f"the claim {c.claim[:60]!r} cites {pid}, which is not a valid quote above"
                )
    return failures


# ---------------------------------------------------------------- prompt and agent


@dataclass(frozen=True)
class Prompt:
    version: int
    system: str
    instructions: str


def load_prompt(path: Path | None = None) -> Prompt:
    """``configs/prompts/evidence_retriever.yaml``; refuses a prompt without an integer version (A4's rule)."""
    path = path or project_root() / PROMPT_FILE
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PromptError(f"cannot read {path}: {exc}") from None
    if not isinstance(raw, dict):
        raise PromptError(f"{path} is not a mapping")
    version = raw.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise PromptError(f"{path} has no version (an integer >= 1)")
    missing = [k for k in ("system", "instructions") if not str(raw.get(k) or "").strip()]
    if missing:
        raise PromptError(f"{path} lacks {missing}")
    return Prompt(version=version, system=raw["system"], instructions=raw["instructions"])


def build_agent(prompt: Prompt) -> Agent[RetrieverDeps, EvidenceAnswer]:
    agent: Agent[RetrieverDeps, EvidenceAnswer] = Agent(
        output_type=EvidenceAnswer,
        deps_type=RetrieverDeps,
        system_prompt=prompt.system,
        instructions=prompt.instructions,
        name=AGENT_NAME,
        tools=[search_passages, get_passage],
        retries=OUTPUT_RETRIES,
        defer_model_check=True,
    )

    @agent.output_validator
    def _validate(ctx: RunContext[RetrieverDeps], answer: EvidenceAnswer) -> EvidenceAnswer:
        failures = validate_answer(answer, ctx.deps.corpus, ctx.deps.ledger)
        if failures:
            ctx.deps.last_failures = failures
            raise ModelRetry(
                "The answer failed these checks; fix them and answer again:\n- "
                + "\n- ".join(failures)
            )
        return answer

    return agent


# ---------------------------------------------------------------- fallback (B9) and the entry point


def _data_as_of(corpus: Corpus) -> str:
    return max(p.valid_to_vintage for p in corpus.passages)


def fallback_answer(question: str, corpus: Corpus, as_of: str | None = None) -> EvidenceAnswer:
    """The deterministic answer from the search alone: the top passages quoted whole, no claims; or an
    abstention with the search's reason."""
    res = bm25_search(corpus, question[:MAX_QUESTION_CHARS], k=FALLBACK_QUOTES, as_of=as_of)
    if res.abstained:
        return EvidenceAnswer(
            abstained=True, abstain_reason=f"the search found no answer: {res.reason}"
        )
    return EvidenceAnswer(
        abstained=False,
        quotes=[Quote(passage_id=h.passage.passage_id, text=h.passage.text) for h in res.hits],
    )


def _evidence_ids(answer: EvidenceAnswer) -> list[str]:
    seen: dict[str, None] = {}
    for q in answer.quotes:
        seen.setdefault(q.passage_id, None)
    return list(seen)


def check_question(question: str) -> str:
    """Input check (B1): 1 to MAX_QUESTION_CHARS characters after trimming; raises ValueError (the API's 400)."""
    q = question.strip()
    if not q or len(q) > MAX_QUESTION_CHARS:
        raise ValueError(f"the question must be 1-{MAX_QUESTION_CHARS} characters")
    return q


def run_evidence_retriever(
    question: str,
    corpus: Corpus,
    model: Model | str,
    as_of: str | None = None,
    prompt: Prompt | None = None,
    agent: Agent[RetrieverDeps, EvidenceAnswer] | None = None,
) -> EvidenceSet:
    """Answer one question. Never raises for a model failure: the fallback answers instead."""
    question = check_question(question)
    prompt = prompt or load_prompt()
    agent = agent or build_agent(prompt)
    deps = RetrieverDeps(corpus=corpus, as_of=as_of)
    fallback = False
    try:
        result = agent.run_sync(
            _user_prompt(question, as_of), deps=deps, model=model, usage_limits=USAGE_LIMITS
        )
        answer = result.output
    except (UnexpectedModelBehavior, AgentRunError):
        answer, fallback = fallback_answer(question, corpus, as_of), True
    return EvidenceSet(
        question=question,
        abstained=answer.abstained,
        abstain_reason=answer.abstain_reason,
        quotes=answer.quotes,
        claims=answer.claims,
        evidence=_evidence_ids(answer),
        corpus_version=corpus.version,
        data_as_of=_data_as_of(corpus),
        as_of=as_of,
        caveat=CAVEAT,
        prompt_version=prompt.version,
        fallback=fallback,
        validator_failures=deps.last_failures,
    )


def _user_prompt(question: str, as_of: str | None) -> str:
    tags = tags_in(question)
    lines = [f"Question: {question}"]
    if tags:
        lines.append(f"Tags named in the question: {', '.join(tags)}")
    if as_of:
        lines.append(f"Use the wordings in force in {as_of}.")
    return "\n".join(lines)


def quoted_tags(es: EvidenceSet) -> Sequence[str]:
    """The tags an answer quotes, first-quoted first (for the golden-set scorer)."""
    seen: dict[str, None] = {}
    for pid in es.evidence:
        seen.setdefault(pid.split(":")[2], None)
    return list(seen)
