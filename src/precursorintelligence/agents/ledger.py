"""The per-run evidence ledger (docs/AGENTS_PLAN.md A3, B3).

Every evidence ID a tool returns in a run goes into that run's ledger; the model may cite only those (validator 1,
``check_evidence_ids``). The runner records every tool result and everything an agent's ``gather`` step fetched, so
no tool has to remember to: the value is walked, and every ``evidence_id`` (a tool record) and every
``evidence_ids`` list (an agent output returned as a tool result: Agent 2 inside Agent 1, a cached brief inside
Agent 4) joins the ledger.

The ledger also keeps each record's month: a record that carries a ``v`` (score, reason, citation, penalty, chain,
portfolio, brief) is from that month, and an answer may cite only records from the month it is labelled with
(``check_evidence_months``). A regulation passage carries no ``v``: it is dated by the vintage its wording starts,
and the corpus version in the cache key already covers it, so it is never month-checked.

Validator 1 also reads the answer's free text: an evidence ID written into a title, a body or a label is checked
like one in an ``evidence_ids`` list (``prose_evidence_ids``).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import BaseModel

from .errors import ModelRetry

EVIDENCE_ID_PREFIXES = (
    "score",
    "reason",
    "citation",
    "penalty",
    "chain",
    "portfolio",
    "brief",
    "passage",
)
"""The evidence ID formats (docs/AGENTS_PLAN.md, the evidence ID table): each is ``<prefix>:<part>:<part>...``."""

_PROSE_ID = re.compile(rf"\b(?:{'|'.join(EVIDENCE_ID_PREFIXES)})(?::[A-Za-z0-9_.-]*[A-Za-z0-9_-])+")
"""An evidence ID written in free text: a known prefix and at least one ``:part`` with no spaces (so "score: 0.31"
isn't one). A trailing full stop is left out."""

ID_FIELDS = frozenset({"evidence_id", "evidence_ids"})


def prose_evidence_ids(value: Any) -> Iterator[str]:
    """Every evidence ID written in the free text of ``value`` (its string fields other than ``evidence_id`` and
    ``evidence_ids``, at any depth)."""
    if isinstance(value, str):
        yield from _PROSE_ID.findall(value)
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            if name not in ID_FIELDS:
                yield from prose_evidence_ids(getattr(value, name))
    elif isinstance(value, Mapping):
        for k, v in value.items():
            if k not in ID_FIELDS:
                yield from prose_evidence_ids(v)
    elif isinstance(value, list | tuple | set | frozenset):
        for v in value:
            yield from prose_evidence_ids(v)


def iter_dated_records(value: Any) -> Iterator[tuple[str, str]]:
    """``(evidence_id, v)`` for every record in ``value`` that carries both, at any depth."""
    if isinstance(value, BaseModel):
        fields = type(value).model_fields
        evidence_id, v = getattr(value, "evidence_id", None), getattr(value, "v", None)
        if (
            "evidence_id" in fields
            and "v" in fields
            and isinstance(evidence_id, str)
            and isinstance(v, str)
        ):
            yield evidence_id, v
        for name in fields:
            yield from iter_dated_records(getattr(value, name))
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from iter_dated_records(v)
    elif isinstance(value, list | tuple | set | frozenset):
        for v in value:
            yield from iter_dated_records(v)


def iter_evidence_ids(value: Any) -> Iterator[str]:
    """Every evidence ID in ``value``: ``evidence_id`` and ``evidence_ids`` fields of models, at any depth."""
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            field = getattr(value, name)
            if name == "evidence_id" and isinstance(field, str):
                yield field
            elif name == "evidence_ids" and isinstance(field, list | tuple):
                yield from (i for i in field if isinstance(i, str))
            else:
                yield from iter_evidence_ids(field)
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from iter_evidence_ids(v)
    elif isinstance(value, list | tuple | set | frozenset):
        for v in value:
            yield from iter_evidence_ids(v)


class EvidenceLedger:
    """The evidence IDs the tools returned in one run."""

    def __init__(self) -> None:
        self._ids: set[str] = set()
        self._months: dict[
            str, str
        ] = {}  # evidence ID -> the month v of its record, for records that carry one

    def record(self, value: Any) -> None:
        self._ids.update(iter_evidence_ids(value))
        self._months.update(iter_dated_records(value))

    def month_of(self, evidence_id: str) -> str | None:
        """The month of the record behind ``evidence_id``, or None when it carries none (a passage, a nested
        answer's ID list)."""
        return self._months.get(evidence_id)

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._ids

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self._ids)


def check_evidence_ids(output: Any, ledger: EvidenceLedger) -> None:
    """Validator 1 (B3): ``ModelRetry`` when the output cites an ID no tool returned in this run, in an ID field or
    written in its text."""
    unknown = sorted({i for i in iter_evidence_ids(output) if i not in ledger})
    in_text = sorted({i for i in prose_evidence_ids(output) if i not in ledger} - set(unknown))
    if unknown or in_text:
        found = ", ".join(
            part
            for part in (
                f"{unknown} were not returned by any tool in this run" if unknown else "",
                f"{in_text} are written in the text but no tool returned them" if in_text else "",
            )
            if part
        )
        raise ModelRetry(
            f"These evidence IDs are not allowed: {found}. Cite only evidence IDs that a tool returned."
        )


def check_evidence_months(output: Any, ledger: EvidenceLedger, month: str) -> None:
    """``ModelRetry`` when the output cites a record from another month than ``month``, the one it is labelled
    with. Only records that carry a ``v`` are checked; a regulation passage is dated by its own vintage."""
    cited = set(iter_evidence_ids(output)) | set(prose_evidence_ids(output))
    other = sorted(i for i in cited if (m := ledger.month_of(i)) is not None and m != month)
    if other:
        raise ModelRetry(
            f"These evidence IDs are from another month than {month}: {other}. Cite only evidence from {month}."
        )
