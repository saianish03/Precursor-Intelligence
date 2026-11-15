"""The chain roll-up (docs/ML_PIPELINE_SPEC.md §10.3, ML-11): facility scores aggregated per ``chain_id``.

No model is trained at chain level: every number is arithmetic on the facilities' calibrated ``p``. Facilities
without a chain get no chain row. ``expected_facilities_with_event`` (sum of p) is exact in expectation; the spread
of a chain's outcomes is wider than independence suggests (§21).
"""

from __future__ import annotations

import polars as pl

CHAIN_COLUMNS = (
    "chain_id",
    "score_month",
    "n_facilities",
    "beds",
    "expected_facilities_with_event",
    "bed_weighted_mean_p",
    "max_p",
    "worst_ccn",
    "share_top_decile",
)


def chain_scores(scores: pl.DataFrame, p_col: str) -> pl.DataFrame:
    """One row per chain from ``scores`` (``ccn, score_month, chain_id, certified_beds, risk_decile`` and the
    calibrated probability ``p_col``).

    ``beds`` sums ``certified_beds`` (nulls count as 0); ``bed_weighted_mean_p`` weights by them over the
    facilities that have a bed count, null when none has. ``worst_ccn`` is the facility with ``max_p`` (the
    smallest ccn on a tie). ``share_top_decile`` is the share of the chain's facilities in decile 1 of the month.
    """
    chained = scores.filter(pl.col("chain_id").is_not_null() & (pl.col("chain_id") != ""))
    beds = pl.col("certified_beds").cast(pl.Float64)
    p = pl.col(p_col)
    has_beds = beds.is_not_null() & (beds > 0)
    bed_sum = pl.when(has_beds).then(beds).otherwise(0.0).sum()
    out = (
        chained.sort([p_col, "ccn"], descending=[True, False])
        .group_by(["chain_id", "score_month"], maintain_order=True)
        .agg(
            pl.len().cast(pl.Int64).alias("n_facilities"),
            beds.fill_null(0.0).sum().alias("beds"),
            p.sum().alias("expected_facilities_with_event"),
            (
                pl.when(bed_sum > 0)
                .then(pl.when(has_beds).then(beds * p).otherwise(0.0).sum() / bed_sum)
                .otherwise(None)
            ).alias("bed_weighted_mean_p"),
            p.first().alias("max_p"),
            pl.col("ccn").first().alias("worst_ccn"),
            (pl.col("risk_decile") == 1).mean().alias("share_top_decile"),
        )
        .sort(["score_month", "chain_id"])
    )
    return out.select(CHAIN_COLUMNS)
