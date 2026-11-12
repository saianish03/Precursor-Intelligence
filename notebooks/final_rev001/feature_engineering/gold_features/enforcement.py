"""``features_enforcement`` (spec §8.4): fines and payment denials, from ``penalties_by_vintage`` at t.

Every penalty row counts, including identical fines (``dup_seq > 1``): they are separate fines (§3.5). Only the
labels collapse rows to one event per ``revision_group``. Provider Info's own fine totals are not used.
"""

from __future__ import annotations

import polars as pl

from ...common.config import FeaturesConfig
from .common import attach, days_since, fill_zero, within
from .reader import FeatureReader

FINE, DENIAL = "Fine", "Payment Denial"
COLUMNS = [
    "n_fines_24m", "fine_dollars_24m", "max_fine_24m", "n_denials_24m", "denial_days_24m",
    "any_penalty_36m", "days_since_last_penalty",
]  # fmt: skip


def build(reader: FeatureReader, rows: pl.DataFrame, t: str, cfg: FeaturesConfig) -> pl.DataFrame:
    penalties = reader.read("penalties_by_vintage", t)
    assert penalties is not None
    p = penalties.filter(pl.col("penalty_date") <= pl.col("cutoff_date"))
    kind, w24 = pl.col("penalty_type"), within("penalty_date", 24)
    fine24, denial24 = (kind == FINE) & w24, (kind == DENIAL) & w24
    per_ccn = p.group_by("ccn").agg(
        n_fines_24m=fine24.sum(),
        fine_sum_24m=pl.col("fine_amount").filter(fine24).sum(),
        max_fine_24m=pl.col("fine_amount").filter(fine24).max(),
        n_denials_24m=denial24.sum(),
        denial_days_24m=pl.col("payment_denial_length_days").filter(denial24).sum(),
        any_penalty_36m=within("penalty_date", 36).any(),
        last_penalty=pl.col("penalty_date").max(),
    )
    return (
        attach(rows, per_ccn)
        .with_columns(
            *fill_zero("n_fines_24m", "fine_sum_24m", "max_fine_24m", "n_denials_24m"),
            *fill_zero("denial_days_24m"),
            pl.col("any_penalty_36m").fill_null(False),
            days_since_last_penalty=days_since(pl.col("last_penalty")),
        )
        .with_columns(
            pl.col("n_fines_24m", "n_denials_24m").cast(pl.Int64),
            pl.col("max_fine_24m", "denial_days_24m").cast(pl.Float64),
            fine_dollars_24m=pl.col("fine_sum_24m").cast(pl.Float64).log1p(),
        )
        .select("ccn", "t", *COLUMNS)
    )
