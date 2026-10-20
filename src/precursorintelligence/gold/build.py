"""Which vintage gold measures its labels from (spec §9.2; SILVER_CONTRACT §8).

Labels are measured from the last vintage that is not a ``stale_refresh`` (``vintage_flags``, written by
bronze); a stale V adds no events. Ledger events first published after that vintage are dropped, so an earlier
V sees only what was published by then.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from . import GoldCheckError

STALE_REFRESH = "stale_refresh"


def label_vintage(
    flags: list[dict[str, Any]], published: list[str], v: str
) -> tuple[str, list[str]]:
    """The last vintage up to ``v`` that is not a stale refresh, and the stale ones skipped on the way.

    Stops (exit 2) instead of guessing when ``vintage_flags`` has no stale_refresh row for a month it needs.
    """
    stale = {r["vintage_month"]: bool(r["value"]) for r in flags if r["flag"] == STALE_REFRESH}
    skipped: list[str] = []
    for month in reversed([p for p in published if p <= v]):
        if month not in stale:
            raise GoldCheckError(
                f"vintage_flags has no stale_refresh row for {month}. Bronze writes it; point --env or "
                "--env-file at the metadata database that came with this silver (e.g. data/ops/metadata.db)"
            )
        if not stale[month]:
            return month, skipped
        skipped.append(month)
    raise GoldCheckError(f"every vintage up to {v} is flagged stale_refresh")


def events_published_by(ledger: pl.DataFrame, v: str) -> pl.DataFrame:
    """Ledger rows already published at vintage ``v`` (an event posted later must not label earlier rows)."""
    return ledger.filter(pl.col("first_seen_vintage") <= v)
