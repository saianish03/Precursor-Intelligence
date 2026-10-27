"""``features_context`` (spec §8.7): the facility against its state at the same t, and the CMS method eras.

Computed within each t from the citations and facility_staffing features, plus any G+ survey in the last 12
months (from ``citations_by_vintage`` at t). A facility without a state gets null state comparisons.
"""

from __future__ import annotations

import polars as pl

from ...common.config import FeaturesConfig
from . import citations, facility_staffing
from .common import attach, gplus_severities, within
from .reader import FeatureReader

COLUMNS = [
    "state_gplus_rate_12m", "state_median_hprd_total", "rel_hprd_total", "rel_sev_points_24m",
    "era", "penalty_regime_2025_10",
]  # fmt: skip


def build(reader: FeatureReader, rows: pl.DataFrame, t: str, cfg: FeaturesConfig) -> pl.DataFrame:
    staffing = facility_staffing.build(reader, rows, t, cfg).select("ccn", "state", "hprd_total")
    cit = citations.build(reader, rows, t, cfg).select("ccn", "sev_points_24m")
    raw = reader.read("citations_by_vintage", t)
    assert raw is not None
    gplus_12m = (
        raw.filter(
            pl.col("scope_severity").is_in(gplus_severities(cfg)),
            within("survey_date", 12),
            pl.col("survey_date") <= pl.col("cutoff_date"),
        )
        .select("ccn")
        .unique()
        .with_columns(gplus_12m=pl.lit(True))
    )
    state = pl.col("state")

    def by_state(expr: pl.Expr) -> pl.Expr:
        return pl.when(state.is_not_null()).then(expr.over("state"))

    median_hprd = by_state(pl.col("hprd_total").median())
    return (
        attach(rows, staffing)
        .pipe(attach, cit)
        .pipe(attach, gplus_12m)
        .with_columns(pl.col("gplus_12m").fill_null(False))
        .with_columns(
            state_gplus_rate_12m=by_state(pl.col("gplus_12m").cast(pl.Float64).mean()),
            state_median_hprd_total=median_hprd,
            rel_hprd_total=pl.when(median_hprd > 0).then(pl.col("hprd_total") / median_hprd),
            rel_sev_points_24m=pl.col("sev_points_24m")
            - by_state(pl.col("sev_points_24m").median()),
            era=pl.lit("pre_2025_07" if t < cfg.rating_method_change else "2025_07_plus"),
            penalty_regime_2025_10=pl.lit(t >= cfg.penalty_regime_change),
        )
        .select("ccn", "t", *COLUMNS)
    )
