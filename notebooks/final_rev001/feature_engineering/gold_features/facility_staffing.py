"""``features_facility_staffing`` (spec §8.5): the facility's profile and staffing, from ``provider_by_vintage`` at
t, and at t−6 / t−12 for the Δ features.

The only gold module that reads the CMS star ratings: ``staffing_rating`` and ``qm_rating`` are ``model``,
``health_inspection_rating`` and ``overall_rating`` ``ablation_only`` (feature_registry.yaml; the column
contract). Δ is null when t−k was never published or the facility is absent from it; no fallback month.
"""

from __future__ import annotations

import polars as pl

from ...common.config import FeaturesConfig
from .common import attach, check_footnotes, footnote_reason
from .reader import FeatureReader

AS_IS = [
    "certified_beds", "ownership_group", "provider_type", "state", "resident_family_council",
    "sprinkler_status", "provider_in_hospital", "ccrc", "abuse_icon", "changed_ownership_12m",
    "most_recent_inspection_over_2y", "hprd_aide", "hprd_lpn", "hprd_rn", "hprd_total",
    "hprd_weekend_total", "hprd_weekend_rn", "hprd_pt", "adj_hprd_rn", "adj_hprd_total",
    "turnover_total", "turnover_rn", "admin_departures", "staffing_rating", "qm_rating",
    "health_inspection_rating", "overall_rating", "chain_id",
]  # fmt: skip
FOOTNOTES = [
    "turnover_total_fn", "turnover_rn_fn", "admin_departures_fn", "reported_staffing_fn",
    "pt_staffing_fn", "staffing_rating_fn", "qm_rating_fn", "avg_residents_per_day_fn",
    "health_inspection_rating_fn", "overall_rating_fn",
]  # fmt: skip
DELTAS = {  # feature -> (column, months back)
    "d6_hprd_total": ("hprd_total", 6),
    "d6_hprd_rn": ("hprd_rn", 6),
    "d12_hprd_total": ("hprd_total", 12),
    "d12_turnover_total": ("turnover_total", 12),
}
COLUMNS = [
    "certified_beds", "occupancy", "ownership_group", "provider_type", "state",
    "resident_family_council", "sprinkler_status", "provider_in_hospital", "ccrc",
    "facility_age_years", "sff_status", "abuse_icon", "changed_ownership_12m",
    "most_recent_inspection_over_2y", "hprd_aide", "hprd_lpn", "hprd_rn", "hprd_total",
    "hprd_weekend_total", "hprd_weekend_rn", "hprd_pt", "adj_hprd_rn", "adj_hprd_total",
    "rn_share", "weekend_gap", "turnover_total", "turnover_rn", "admin_departures", *FOOTNOTES,
    "staffing_rating", "qm_rating", "health_inspection_rating", "overall_rating", "chain_id",
    *DELTAS,
]  # fmt: skip
DAYS_PER_YEAR = 365.25


def _delta(reader: FeatureReader, t: str, feature: str, col: str, months: int) -> pl.DataFrame:
    before = reader.read("provider_by_vintage", t, months_back=months)
    if before is None:  # never published: every Δ is null
        return pl.DataFrame(schema={"ccn": pl.String, f"_{feature}": pl.Float64})
    return before.select("ccn", pl.col(col).cast(pl.Float64).alias(f"_{feature}"))


def build(reader: FeatureReader, rows: pl.DataFrame, t: str, cfg: FeaturesConfig) -> pl.DataFrame:
    provider = reader.read("provider_by_vintage", t)
    assert provider is not None
    check_footnotes(provider, FOOTNOTES, cfg, "provider_by_vintage")
    beds = pl.col("certified_beds")
    profile = provider.select(
        "ccn",
        *AS_IS,
        *(footnote_reason(c, cfg) for c in FOOTNOTES),
        occupancy=pl.when(beds > 0).then(
            (pl.col("avg_residents_per_day") / beds).clip(upper_bound=cfg.occupancy_cap)
        ),
        facility_age_years=(pl.col("cutoff_date") - pl.col("date_first_approved")).dt.total_days()
        / DAYS_PER_YEAR,
        sff_status=pl.col("special_focus_status"),
        rn_share=pl.when(pl.col("hprd_total") > 0).then(pl.col("hprd_rn") / pl.col("hprd_total")),
        weekend_gap=pl.col("hprd_total") - pl.col("hprd_weekend_total"),
    )
    out = attach(rows, profile)
    for feature, (col, months) in DELTAS.items():
        out = attach(out, _delta(reader, t, feature, col, months)).with_columns(
            (pl.col(col) - pl.col(f"_{feature}")).alias(feature)
        )
    return out.select("ccn", "t", *COLUMNS)
