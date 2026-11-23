"""The in-memory daily counter (docs/AGENTS_PLAN.md D3's interface)."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from precursorintelligence.agents.counter import DailyCapReached, InMemoryDailyCounter

DAY = date(2026, 11, 21)


def test_concurrent_runs_never_pass_the_cap():
    counter = InMemoryDailyCounter(run_cap=3, spend_stop_usd=2.0)

    async def go() -> list[object]:
        return await asyncio.gather(
            *(counter.reserve_run(DAY) for _ in range(10)), return_exceptions=True
        )

    results = asyncio.run(go())
    assert sum(not isinstance(r, Exception) for r in results) == 3
    assert all(isinstance(r, DailyCapReached) for r in results if isinstance(r, Exception))
    assert asyncio.run(counter.usage(DAY)).runs == 3


def test_runs_and_cost_are_kept_per_day_and_a_release_gives_a_run_back():
    counter = InMemoryDailyCounter(run_cap=1, spend_stop_usd=0.05)
    asyncio.run(counter.reserve_run(DAY))
    asyncio.run(counter.add_cost(DAY, 0.01))
    with pytest.raises(DailyCapReached):
        asyncio.run(counter.reserve_run(DAY))
    asyncio.run(counter.reserve_run(date(2026, 11, 22)))  # a new day starts empty
    asyncio.run(counter.release_run(DAY))
    u = asyncio.run(counter.reserve_run(DAY))
    assert (u.runs, u.cost_usd) == (1, pytest.approx(0.01))


def test_a_negative_cost_is_refused():
    with pytest.raises(ValueError):
        asyncio.run(InMemoryDailyCounter(run_cap=1, spend_stop_usd=1).add_cost(DAY, -1))
