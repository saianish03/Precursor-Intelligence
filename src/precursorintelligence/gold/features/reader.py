"""The only way feature code reads silver (spec §8, §11.3; SILVER_CONTRACT §6).

A feature for row (ccn, t) may read a ``*_by_vintage`` table or ``citation_lookup`` at vintage t, and change
(Δ) features also at t−6 and t−12. The reader refuses every other table (ledgers, ``facility_presence``,
``citation_lookup_versions``, ``dim_facility_latest`` ...), any other month offset, and, when built with
``limit``, any vintage after it. It records every read, so tests can check what a build touched, and a reader
limited to t is the in-gate time-travel test: features rebuilt through it must equal the stored ones.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import polars as pl

from ...common.io.object_store import ObjectStore
from .. import GoldCheckError

FEATURE_INPUTS = (
    "provider_by_vintage",
    "citations_by_vintage",
    "surveys_by_vintage",
    "penalties_by_vintage",
    "mds_qm_by_vintage",
    "claims_qm_by_vintage",
    "citation_lookup",
)
MONTHS_BACK = (0, 6, 12)  # t, and t−6 / t−12 for Δ features


class FeatureReadError(GoldCheckError):
    """Feature code asked for something it may not read."""


def shift_month(t: str, months: int) -> str:
    """``t`` (YYYY-MM) plus ``months`` (may be negative)."""
    y, m = map(int, t.split("-"))
    n = y * 12 + m - 1 + months
    return f"{n // 12:04d}-{n % 12 + 1:02d}"


class FeatureReader:
    """Reads one silver partition at a time through ``load(table, vintage)``.

    ``published`` are the vintages that exist (the silver provider partitions); asking for one that was never
    published returns None, which Δ features turn into null.
    """

    def __init__(
        self,
        load: Callable[[str, str], pl.DataFrame],
        *,
        published: Sequence[str],
        limit: str | None = None,
    ) -> None:
        self._load = load
        self.published = sorted(published)
        self.limit = limit
        self.reads: list[tuple[str, str]] = []

    def read(self, table: str, t: str, *, months_back: int = 0) -> pl.DataFrame | None:
        """``table`` at vintage t − ``months_back``; None if that vintage was never published."""
        if table not in FEATURE_INPUTS:
            raise FeatureReadError(
                f"feature code may not read {table!r} (allowed: {FEATURE_INPUTS})"
            )
        if months_back not in MONTHS_BACK:
            raise FeatureReadError(f"feature code may read t, t-6 or t-12, not t-{months_back}")
        vintage = shift_month(t, -months_back)
        if self.limit is not None and max(t, vintage) > self.limit:
            raise FeatureReadError(f"read of {table} for t={t} goes past the limit {self.limit}")
        self.reads.append((table, vintage))
        if vintage not in self.published:
            return None
        return self._load(table, vintage)

    def limited_to(self, t: str) -> FeatureReader:
        """A reader for the rows of month t: limit = min(this limit, t). It shares the loader and the read log,
        so a family that reads past its own t fails in every run, not only in the time-travel test."""
        limit = t if self.limit is None else min(self.limit, t)
        child = FeatureReader(self._load, published=self.published, limit=limit)
        child.reads = self.reads
        return child

    @classmethod
    def from_frames(
        cls, silver: dict[str, pl.DataFrame], *, limit: str | None = None
    ) -> FeatureReader:
        """Over stacked in-memory tables (all vintages, with ``vintage_month``), e.g. the toy world."""

        def load(table: str, vintage: str) -> pl.DataFrame:
            if table not in silver:
                raise GoldCheckError(f"missing silver input: {table}")
            return silver[table].filter(pl.col("vintage_month") == vintage)

        published = silver["provider_by_vintage"]["vintage_month"].unique().to_list()
        return cls(load, published=published, limit=limit)

    @classmethod
    def from_store(
        cls, store: ObjectStore, *, published: Sequence[str], limit: str | None = None
    ) -> FeatureReader:
        """Over the silver partitions in an object store (``silver/<table>/vintage=<v>/part-0.parquet``)."""

        def load(table: str, vintage: str) -> pl.DataFrame:
            key = f"silver/{table}/vintage={vintage}/part-0.parquet"
            if not store.exists(key):
                raise GoldCheckError(f"missing silver input: {key}")
            with store.open_read(key) as f:
                return pl.read_parquet(f)

        return cls(load, published=published, limit=limit)
