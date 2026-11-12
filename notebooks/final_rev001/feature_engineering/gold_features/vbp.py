"""``features_vbp`` (docs/ALL_TABLES_CONTRACT.md): the SNF Value-Based Purchasing result, from ``vbp_by_vintage``
at t.

CMS published the VBP facility file only until 2026-06, so the read is optional: a month without it gets
``vbp_available`` False and null values. A facility missing from a published file gets nulls with
``vbp_available`` True. The rank becomes a percentile within t (rank / max rank; small = better), comparable
across fiscal years whose number of ranked facilities differs. Measures absent in a fiscal year stay null.
"""

from __future__ import annotations

import polars as pl

from ...common.config import FeaturesConfig
from .common import attach
from .reader import FeatureReader

AS_IS = {  # feature -> silver column
    "vbp_performance_score": "performance_score",
    "vbp_incentive_multiplier": "incentive_multiplier",
    "vbp_snfrm_performance_rate": "snfrm_performance_rate",
    "vbp_hai_performance_rate": "hai_performance_rate",
    "vbp_turnover_performance_rate": "turnover_performance_rate",
    "vbp_staffing_performance_hprd": "staffing_performance_hprd",
}
COLUMNS = [
    "vbp_available", *AS_IS, "vbp_rank_pct", "vbp_snfrm_change", "vbp_penalized",
]  # fmt: skip


def build(reader: FeatureReader, rows: pl.DataFrame, t: str, cfg: FeaturesConfig) -> pl.DataFrame:
    vbp = reader.read("vbp_by_vintage", t, optional=True)
    if vbp is None:
        nulls = [pl.lit(None, pl.Float64).alias(c) for c in COLUMNS[1:-1]]
        return rows.select(
            "ccn",
            "t",
            pl.lit(False).alias("vbp_available"),
            *nulls,
            pl.lit(None, pl.Boolean).alias("vbp_penalized"),
        ).select("ccn", "t", *COLUMNS)
    max_rank = pl.col("vbp_rank").cast(pl.Float64).max()
    per_ccn = vbp.select(
        "ccn",
        *(pl.col(src).cast(pl.Float64).alias(f) for f, src in AS_IS.items()),
        vbp_rank_pct=pl.when(max_rank > 0).then(pl.col("vbp_rank").cast(pl.Float64) / max_rank),
        vbp_snfrm_change=(pl.col("snfrm_performance_rate") - pl.col("snfrm_baseline_rate")).cast(
            pl.Float64
        ),
        vbp_penalized=pl.col("incentive_multiplier") < 1,
    )
    return (
        attach(rows, per_ccn).with_columns(vbp_available=pl.lit(True)).select("ccn", "t", *COLUMNS)
    )
