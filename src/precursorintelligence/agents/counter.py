"""The daily call cap and spend stop (docs/AGENTS_PLAN.md D3): the interface and an in-memory version.

One counter per UTC day holds the runs started and their estimated cost. A run is reserved before the model is
called, and refused when the day is at the run cap or the spend stop. Nested runs (Agent 2 called by Agent 1) reserve
their own run. A cache hit reserves nothing; a run that ends in the fallback answer gives its run back (its cost
stays, because its model calls were made). The cost can overshoot the stop by the runs already in flight.

The GCS version (D3) implements the same interface with a generation-match write, so two instances can't race.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol


@dataclass(frozen=True)
class DailyUsage:
    day: date
    runs: int
    cost_usd: float


class DailyCapReached(Exception):
    """The day's run cap or spend stop is reached; the API answers 429 (D3)."""

    def __init__(self, reason: Literal["runs", "spend"], usage: DailyUsage) -> None:
        self.reason = reason
        self.usage = usage
        what = (
            f"the daily cap of runs is reached ({usage.runs} today)"
            if reason == "runs"
            else f"the daily spend stop is reached (${usage.cost_usd:.2f} today)"
        )
        super().__init__(f"{what}; try again tomorrow (UTC)")


class DailyCounter(Protocol):
    async def reserve_run(self, day: date) -> DailyUsage:
        """Count one run for ``day``, or raise ``DailyCapReached`` (and count nothing) when the day is full."""
        ...

    async def release_run(self, day: date) -> DailyUsage:
        """Give back a run reserved for ``day`` (a run that ended in the fallback answer)."""
        ...

    async def add_cost(self, day: date, cost_usd: float) -> DailyUsage:
        """Add a run's estimated cost to ``day``."""
        ...

    async def usage(self, day: date) -> DailyUsage: ...


class InMemoryDailyCounter:
    """``DailyCounter`` in a dict, for tests and local runs. Not shared across instances."""

    def __init__(self, *, run_cap: int, spend_stop_usd: float) -> None:
        self._run_cap = run_cap
        self._spend_stop_usd = spend_stop_usd
        self._days: dict[date, DailyUsage] = {}
        self._lock = asyncio.Lock()

    def _get(self, day: date) -> DailyUsage:
        return self._days.get(day, DailyUsage(day, 0, 0.0))

    async def reserve_run(self, day: date) -> DailyUsage:
        async with self._lock:
            u = self._get(day)
            if u.runs >= self._run_cap:
                raise DailyCapReached("runs", u)
            if u.cost_usd >= self._spend_stop_usd:
                raise DailyCapReached("spend", u)
            self._days[day] = u = DailyUsage(day, u.runs + 1, u.cost_usd)
            return u

    async def release_run(self, day: date) -> DailyUsage:
        async with self._lock:
            u = self._get(day)
            self._days[day] = u = DailyUsage(day, max(u.runs - 1, 0), u.cost_usd)
            return u

    async def add_cost(self, day: date, cost_usd: float) -> DailyUsage:
        if cost_usd < 0:
            raise ValueError(f"a cost can't be negative: {cost_usd}")
        async with self._lock:
            u = self._get(day)
            self._days[day] = u = DailyUsage(day, u.runs, u.cost_usd + cost_usd)
            return u

    async def usage(self, day: date) -> DailyUsage:
        async with self._lock:
            return self._get(day)
