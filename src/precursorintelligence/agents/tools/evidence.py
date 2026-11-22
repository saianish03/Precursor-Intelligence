"""The evidence tools of Agents 1 and 4 (docs/AGENTS_PLAN.md E3, E4).

Read-only, typed and bounded by a timeout. Every record carries a stable evidence ID, the month ``v`` and the
``gold_run_id`` it came from, so the validators can check each ID and number an agent writes against the record.
Counting is done here in polars, never by the model.

Evidence IDs (stable within a month's evidence):

- ``score:<ccn>:<v>``
- ``reason:<ccn>:<v>:<raises|lowers>:<rank>``: the rank restarts at 1 in each direction
- ``citation:<ccn>:<survey_date>:<ftag>``: the citation key (one prefix, F)
- ``penalty:<ccn>:<penalty_date>:<fine|payment_denial>:<n>``: one month can list several penalties of one type on
  one date (identical rows are distinct fines), numbered from 1 in a fixed order
- ``chain:<chain_id>:<v>``
- ``portfolio:<v>:<sha256 of the sorted CCNs and top_n, first 12 hex>``

A repeated ID within one tool's answer refuses (``EvidenceError``) rather than give a validator two records for one
ID.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any, Literal, TypeVar

import polars as pl
from pydantic import BaseModel, ConfigDict

from .source import EvidenceError, EvidenceSource

CCN_PATTERN = re.compile(r"^[0-9A-Z]{6}$")  # as harmonize/normalize.py CCN_PATTERN
DECILES = range(1, 11)

T = TypeVar("T")


class ToolInputError(ValueError):
    """An argument a tool cannot use (e.g. a malformed CCN)."""


class ToolTimeout(TimeoutError):
    """A tool did not answer within its timeout."""


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    v: str
    gold_run_id: str


class FacilityScore(Record):
    ccn: str
    p: float
    risk_percentile: float
    risk_decile: int
    state: str | None
    chain_id: str | None
    model_version: int


class Reason(Record):
    ccn: str
    direction: Literal["raises", "lowers"]
    rank: int
    feature: str
    label: str
    feature_value: str | None
    contribution_logodds: float


class Citation(Record):
    ccn: str
    survey_date: date
    ftag: str
    deficiency_category: str
    scope_severity: str
    is_standard: bool
    is_complaint: bool
    correction_date: date | None


class Penalty(Record):
    ccn: str
    penalty_date: date
    penalty_type: str
    fine_amount: float | None
    payment_denial_start_date: date | None
    payment_denial_length_days: float | None


class Chain(Record):
    chain_id: str
    n_facilities: int
    beds: float
    expected_facilities_with_event: float
    p_any_event: float
    bed_weighted_mean_p: float | None
    max_p: float
    worst_ccn: str
    share_top_decile: float
    model_version: int


class PortfolioStats(Record):
    n_requested: int
    n_scored: int
    unscored_ccns: list[str]
    decile_counts: dict[int, int]  # every decile 1-10, 1 = the riskiest tenth of the month
    mean_p: float | None
    expected_facilities_with_event: float
    state_counts: dict[str, int]  # "unknown" for a facility without a state
    chain_counts: dict[
        str, int
    ]  # facilities per chain, most first; unchained facilities are left out
    top: list[FacilityScore]  # the highest p first


def _fields(model: type[Record], row: dict[str, Any]) -> dict[str, Any]:
    """The row's values for ``model``; the stamp (ID, month, gold run) comes from the manifest, which the source
    has checked every row against."""
    return {k: row[k] for k in model.model_fields if k in row and k not in Record.model_fields}


R = TypeVar("R", bound=Record)


def _unique_ids(records: list[R]) -> list[R]:
    ids = [r.evidence_id for r in records]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise EvidenceError(f"the evidence repeats the IDs {dup}")
    return records


def normalize_ccn(ccn: str) -> str:
    """``ccn`` stripped and upper-cased; ``ToolInputError`` unless it is 6 characters of 0-9 and A-Z."""
    value = ccn.strip().upper()
    if not CCN_PATTERN.match(value):
        raise ToolInputError(f"not a CCN (6 characters, 0-9 and A-Z): {ccn!r}")
    return value


class EvidenceTools:
    """The tools over one month's ``EvidenceSource``. Each call gives up after ``timeout_s`` seconds."""

    def __init__(self, source: EvidenceSource, *, timeout_s: float) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive: {timeout_s}")
        self._source = source
        self._timeout_s = timeout_s
        m = source.manifest
        self._v, self._gold_run_id = m.v, m.gold_run_id

    async def get_facility_score(self, ccn: str) -> FacilityScore | None:
        """The facility's calibrated probability, percentile and decile, or None when it was not scored."""
        return await self._run(self._facility_score, normalize_ccn(ccn))

    async def get_reasons(self, ccn: str) -> list[Reason]:
        """The facility's top drivers: those that raise its risk, then those that lower it, by rank."""
        return await self._run(self._reasons, normalize_ccn(ccn))

    async def get_citations(self, ccn: str) -> list[Citation]:
        """The facility's health citations in the evidence, newest first."""
        return await self._run(self._citations, normalize_ccn(ccn))

    async def get_penalties(self, ccn: str) -> list[Penalty]:
        """The facility's fines and payment denials in the evidence, newest first."""
        return await self._run(self._penalties, normalize_ccn(ccn))

    async def get_chain(self, chain_id: str) -> Chain | None:
        """The chain roll-up of ``chain_id``, or None when no scored facility belongs to it."""
        value = chain_id.strip()
        if not value or len(value) > 64:
            raise ToolInputError(f"not a chain id: {chain_id!r}")
        return await self._run(self._chain, value)

    async def portfolio_stats(self, ccns: Sequence[str], *, top_n: int) -> PortfolioStats:
        """Counts over a book of facilities (duplicates count once) and its ``top_n`` riskiest."""
        if top_n < 0:
            raise ToolInputError(f"top_n must not be negative: {top_n}")
        book = sorted({normalize_ccn(c) for c in ccns})
        return await self._run(self._portfolio, book, top_n)

    # ------------------------------------------------------------------ the reads, run in a worker thread

    async def _run(self, fn: Callable[..., T], *args: object) -> T:
        try:
            return await asyncio.wait_for(asyncio.to_thread(fn, *args), self._timeout_s)
        except TimeoutError:
            raise ToolTimeout(f"{fn.__name__.lstrip('_')} took over {self._timeout_s}s") from None

    def _stamp(self, evidence_id: str) -> dict[str, Any]:
        return {"evidence_id": evidence_id, "v": self._v, "gold_run_id": self._gold_run_id}

    def _score_record(self, row: dict[str, Any]) -> FacilityScore:
        return FacilityScore(
            **_fields(FacilityScore, row), **self._stamp(f"score:{row['ccn']}:{self._v}")
        )

    def _facility_score(self, ccn: str) -> FacilityScore | None:
        df = self._source.rows("facility_scores", "ccn", [ccn])
        if df.height > 1:
            raise EvidenceError(f"facility_scores has {df.height} rows for {ccn}")
        return self._score_record(df.row(0, named=True)) if df.height else None

    def _reasons(self, ccn: str) -> list[Reason]:
        df = self._source.rows("facility_reasons", "ccn", [ccn]).sort(
            ["direction", "rank"],
            descending=[True, False],  # raises before lowers
        )
        return _unique_ids(
            [
                Reason(
                    **_fields(Reason, r),
                    **self._stamp(f"reason:{ccn}:{self._v}:{r['direction']}:{r['rank']}"),
                )
                for r in df.iter_rows(named=True)
            ]
        )

    def _citations(self, ccn: str) -> list[Citation]:
        df = self._source.rows("citations", "ccn", [ccn]).sort(
            ["survey_date", "ftag"], descending=[True, False]
        )
        return _unique_ids(
            [
                Citation(
                    **_fields(Citation, r),
                    **self._stamp(f"citation:{ccn}:{r['survey_date'].isoformat()}:{r['ftag']}"),
                )
                for r in df.iter_rows(named=True)
            ]
        )

    def _penalties(self, ccn: str) -> list[Penalty]:
        group = ["penalty_date", "penalty_type"]
        df = (
            self._source.rows("penalties", "ccn", [ccn])
            .sort([*group, "fine_amount", "payment_denial_start_date", "dup_seq"], nulls_last=True)
            .with_columns(pl.int_range(1, pl.len() + 1).over(group).alias("_n"))
            .sort(["penalty_date", "penalty_type", "_n"], descending=[True, False, False])
        )
        out = []
        for r in df.iter_rows(named=True):
            kind = str(r["penalty_type"]).lower().replace(" ", "_")
            pid = f"penalty:{ccn}:{r['penalty_date'].isoformat()}:{kind}:{r['_n']}"
            out.append(Penalty(**_fields(Penalty, r), **self._stamp(pid)))
        return _unique_ids(out)

    def _chain(self, chain_id: str) -> Chain | None:
        df = self._source.rows("chain_scores", "chain_id", [chain_id])
        if df.height > 1:
            raise EvidenceError(f"chain_scores has {df.height} rows for chain {chain_id}")
        if not df.height:
            return None
        r = df.row(0, named=True)
        return Chain(
            **_fields(Chain, r),
            **self._stamp(f"chain:{chain_id}:{self._v}"),
        )

    def _portfolio(self, book: list[str], top_n: int) -> PortfolioStats:
        scores = self._source.rows("facility_scores", "ccn", book)
        if scores["ccn"].n_unique() != scores.height:
            raise EvidenceError("facility_scores has more than one row for a CCN of the book")
        scored = set(scores["ccn"].to_list())
        deciles = dict(scores.group_by("risk_decile").len().iter_rows())
        states = (
            scores.select(pl.col("state").fill_null("unknown"))
            .group_by("state")
            .len()
            .sort(["len", "state"], descending=[True, False])
        )
        chains = (
            scores.filter(pl.col("chain_id").is_not_null() & (pl.col("chain_id") != ""))
            .group_by("chain_id")
            .len()
            .sort(["len", "chain_id"], descending=[True, False])
        )
        top = scores.sort(["p", "ccn"], descending=[True, False]).head(top_n)
        digest = hashlib.sha256(f"{','.join(book)}|{top_n}".encode()).hexdigest()[:12]
        return PortfolioStats(
            n_requested=len(book),
            n_scored=scores.height,
            unscored_ccns=[c for c in book if c not in scored],
            decile_counts={d: int(deciles.get(d, 0)) for d in DECILES},
            mean_p=float(scores["p"].mean()) if scores.height else None,  # type: ignore[arg-type]
            expected_facilities_with_event=float(scores["p"].sum()),
            state_counts=dict(states.iter_rows()),
            chain_counts=dict(chains.iter_rows()),
            top=[self._score_record(r) for r in top.iter_rows(named=True)],
            **self._stamp(f"portfolio:{self._v}:{digest}"),
        )
