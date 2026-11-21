"""The per-run evidence ledger (docs/AGENTS_PLAN.md A3, B3).

Every evidence ID a tool returns in a run goes into that run's ledger; the model may cite only those (validator 1,
``check_evidence_ids``). ``LedgerToolset`` wraps an agent's tools so nothing has to remember to record: whatever a
tool returns is walked, and every ``evidence_id`` (a tool record) and every ``evidence_ids`` list (an agent output
returned as a tool result: Agent 2 inside Agent 1, a cached brief inside Agent 4) joins the ledger.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import BaseModel
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset
from pydantic_ai.toolsets.abstract import ToolsetTool


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

    def record(self, value: Any) -> None:
        self._ids.update(iter_evidence_ids(value))

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._ids

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self._ids)


class LedgerToolset(WrapperToolset[Any]):
    """Records every tool result in ``ctx.deps.ledger`` before the model sees it."""

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: ToolsetTool[Any]
    ) -> Any:
        result = await super().call_tool(name, tool_args, ctx, tool)
        ctx.deps.ledger.record(result)
        return result


def with_ledger(toolset: AbstractToolset[Any]) -> LedgerToolset:
    return LedgerToolset(wrapped=toolset)


def check_evidence_ids(output: Any, ledger: EvidenceLedger) -> None:
    """Validator 1 (B3): ``ModelRetry`` when the output cites an ID no tool returned in this run."""
    unknown = sorted({i for i in iter_evidence_ids(output) if i not in ledger})
    if unknown:
        raise ModelRetry(
            f"These evidence IDs were not returned by any tool in this run: {unknown}. "
            "Cite only evidence IDs that a tool returned."
        )
