"""Step 6 (spec §7): the spine, one row per (ccn, t) for every published t from spine_start to V.

Every later gold table has exactly one row per spine row and is joined onto it by (ccn, t).
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from . import GoldCheckError

SPINE_COLUMNS = ["ccn", "t", "cutoff_date"]


def build_spine(
    facility_universe: pl.DataFrame,
    *,
    published_vintages: Sequence[str],
    spine_start: str,
    v: str,
) -> pl.DataFrame:
    """Rows for every CCN in ``facility_universe`` at each t in [spine_start, v].

    ``published_vintages`` lists the vintages that exist; never-published months simply have no rows.
    """
    spine = (
        facility_universe.filter(pl.col("vintage_month").is_between(pl.lit(spine_start), pl.lit(v)))
        .select("ccn", t=pl.col("vintage_month"))
        .with_columns(cutoff_date=(pl.col("t") + "-01").str.to_date("%Y-%m-%d"))
        .sort("t", "ccn")
    )
    if spine.select("ccn", "t").is_duplicated().any():
        raise GoldCheckError("spine: (ccn, t) is not unique")
    expected = {t for t in published_vintages if spine_start <= t <= v}
    unpublished = sorted(set(spine["t"]) - expected)
    if unpublished:
        raise GoldCheckError(f"spine: rows for months that were never published: {unpublished}")
    missing = sorted(expected - set(spine["t"]))
    if missing:
        raise GoldCheckError(f"spine: published months without rows: {missing}")
    return spine
