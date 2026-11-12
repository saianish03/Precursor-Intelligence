"""``features_survey_summary`` (docs/ALL_TABLES_CONTRACT.md): the deficiency counts of the last three standard
inspection cycles, from ``survey_summary_by_vintage`` at t (one row per ccn x inspection_cycle; 1 = latest).

A cycle's health counts count only when its health survey date is on or before the cutoff, and its fire counts
only when its fire survey date is; otherwise that side of the cycle is treated as not on record.
"""

from __future__ import annotations

import polars as pl

from ...common.config import FeaturesConfig
from .common import CUTOFF, attach, days_since, fill_zero
from .reader import FeatureReader

# cycle-1 health deficiency categories (silver names, ALL_TABLES_CONTRACT); the rest overlap features_citations
HEALTH_CATEGORIES = ("n_def_abuse_neglect", "n_def_quality_of_life_care", "n_def_infection_control")
COLUMNS = [
    "c1_n_health_def", "c1_n_fire_def", *(f"c1_{c}" for c in HEALTH_CATEGORIES),
    "c23_mean_health_def", "health_def_trend", "c23_mean_fire_def", "fire_def_trend",
    "max_health_def_3cycles", "days_since_c1_health_survey", "days_c1_to_c2_health_survey",
    "days_between_health_fire_c1",
]  # fmt: skip
COUNTS = [
    "c1_n_health_def", "c1_n_fire_def", *(f"c1_{c}" for c in HEALTH_CATEGORIES), "max_health_def_3cycles",
]  # fmt: skip


def build(reader: FeatureReader, rows: pl.DataFrame, t: str, cfg: FeaturesConfig) -> pl.DataFrame:
    summary = reader.read("survey_summary_by_vintage", t)
    assert summary is not None  # t is a spine month, so it was published
    health_seen = pl.col("health_survey_date") <= CUTOFF  # null date -> null -> not on record
    fire_seen = pl.col("fire_survey_date") <= CUTOFF
    s = summary.with_columns(
        pl.when(health_seen).then(pl.col("health_survey_date")).alias("health_date"),
        pl.when(fire_seen).then(pl.col("fire_survey_date")).alias("fire_date"),
        *(
            pl.when(health_seen).then(pl.col(c)).alias(c)
            for c in ("n_health_def", *HEALTH_CATEGORIES)
        ),
        pl.when(fire_seen).then(pl.col("n_fire_def")).alias("n_fire_def"),
    )
    cycle = pl.col("inspection_cycle")
    c1, c23 = cycle == 1, cycle.is_in([2, 3])

    def on(cond: pl.Expr, col: str) -> pl.Expr:
        return pl.col(col).filter(cond).first()

    per_ccn = s.group_by("ccn").agg(
        c1_n_health_def=on(c1, "n_health_def"),
        c1_n_fire_def=on(c1, "n_fire_def"),
        **{f"c1_{c}": on(c1, c) for c in HEALTH_CATEGORIES},
        c23_mean_health_def=pl.col("n_health_def").filter(c23).mean().cast(pl.Float64),
        c23_mean_fire_def=pl.col("n_fire_def").filter(c23).mean().cast(pl.Float64),
        max_health_def_3cycles=pl.col("n_health_def").max(),
        c1_health=on(c1, "health_date"),
        c2_health=on(cycle == 2, "health_date"),
        c1_fire=on(c1, "fire_date"),
    )
    days = (pl.col("c1_health") - pl.col("c2_health")).dt.total_days().cast(pl.Float64)
    gap = (pl.col("c1_health") - pl.col("c1_fire")).dt.total_days().abs().cast(pl.Float64)
    return (
        attach(rows, per_ccn)
        .with_columns(
            # trends use the raw cycle-1 value: null when cycle 1 or both older cycles are not on record
            health_def_trend=(pl.col("c1_n_health_def") - pl.col("c23_mean_health_def")),
            fire_def_trend=(pl.col("c1_n_fire_def") - pl.col("c23_mean_fire_def")),
            days_since_c1_health_survey=days_since(pl.col("c1_health")),
            days_c1_to_c2_health_survey=days,
            days_between_health_fire_c1=gap,
        )
        .with_columns(*fill_zero(*COUNTS))
        .with_columns(pl.col(*COUNTS).cast(pl.Int64))
        .select("ccn", "t", *COLUMNS)
    )
