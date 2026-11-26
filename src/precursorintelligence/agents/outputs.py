"""The shared output models every agent's answer is built from (docs/AGENTS_PLAN.md A5).

The evidence itself is Riya's ``Record`` (``agents/tools/evidence.py``: ``evidence_id``, ``v``, ``gold_run_id``);
there is no second evidence type. An output cites records by their evidence IDs, and validator 1 checks every one
against the run's ledger (B3), wherever it sits: ``AgentOutput.evidence_ids``, a ``Driver.evidence_id`` or a
``Section.evidence_ids``.

``data_as_of``, ``caveat`` and ``fallback`` are never written by the model: they are left out of the schema it sees,
and ``run_agent`` stamps them on every answer, the fallback and a cache hit included. Both are per agent: the
caveat stamped is the output type's own ``caveat`` default (``CAVEAT`` unless the agent overrides it, as Agent 2's
does), and ``data_as_of`` comes from ``AgentDefinition.data_as_of`` (by default the score month ``v``). The API
wraps ``data_as_of`` in the full data-version object (``v``, ``score_month``, ``model_version``, ``gold_run_id``)
from the evidence manifest, so this model keeps the plain month.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from .tools.evidence import Reason, Record

CAVEAT = "Regulatory risk, not a claims prediction."
"""The score caveat, kept in this one constant; its wording is still to be agreed with the frontend (PRODUCT.md)."""


class Driver(BaseModel):
    """One driver of a facility's score, citing its ``reason:<ccn>:<v>:<raises|lowers>:<rank>`` record. Its
    direction must match the record's SHAP sign (validator 3, B5)."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    direction: Literal["raises", "lowers"]
    evidence_id: str = Field(min_length=1)

    @classmethod
    def from_reason(cls, reason: Reason) -> Driver:
        """The driver a ``Reason`` record states, for the fallback templates."""
        return cls(label=reason.label, direction=reason.direction, evidence_id=reason.evidence_id)


class Section(BaseModel):
    """One titled part of an answer and the evidence it rests on."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class AgentOutput(BaseModel):
    """The base of every agent's output type (``InvestigationBrief``, ``EvidenceSet``, ``OpsReport``,
    ``ExecutiveBrief``): the evidence list, the data month and the caveat."""

    model_config = ConfigDict(extra="forbid")

    evidence_ids: list[str] = Field(default_factory=list)
    # the data month (by default the score month v); stamped by run_agent
    data_as_of: SkipJsonSchema[str | None] = None
    # this default is what run_agent stamps: an agent with its own caveat overrides it in its output type
    caveat: SkipJsonSchema[str] = CAVEAT
    # True when the template answered (B9); stamped by run_agent
    fallback: SkipJsonSchema[bool] = False


def evidence_ids_of(records: Iterable[Record]) -> list[str]:
    """The evidence IDs of ``records``, in order, each once: for the fallback templates."""
    return list(dict.fromkeys(r.evidence_id for r in records))
