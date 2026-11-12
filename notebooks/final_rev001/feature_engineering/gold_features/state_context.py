"""``features_state_context`` (docs/ALL_TABLES_CONTRACT.md): the facility against its state at t, from
``provider_by_vintage`` (the facility's own value and state) and ``state_averages_by_vintage`` (one row per state).

Differences are facility minus state average; ratios are facility / state average, null when the state average
is null or not positive. The ``NATION`` row is never a state. A facility absent from Provider Info at t, or whose
state has no row, gets nulls. Only the columns named here are read: the state rating columns
(``st_overall_rating`` ...) never become features (star ratings are never features).

The state file has no adjusted (case-mix) staffing, so ``adj_hprd_total_vs_state`` compares the facility's
adjusted total to the state's reported total.
"""

from __future__ import annotations

import polars as pl

from ...common.config import FeaturesConfig
from .common import attach
from .reader import FeatureReader

NATION = "NATION"
# feature -> (facility column in provider_by_vintage, state column in state_averages_by_vintage)
DIFFS = {
    "hprd_total_vs_state": ("hprd_total", "st_hprd_total"),
    "hprd_rn_vs_state": ("hprd_rn", "st_hprd_rn"),
    "adj_hprd_total_vs_state": ("adj_hprd_total", "st_hprd_total"),
    "turnover_total_vs_state": ("turnover_total", "st_turnover_total"),
    "turnover_rn_vs_state": ("turnover_rn", "st_turnover_rn"),
}
RATIOS = {
    "hprd_total_ratio_state": ("hprd_total", "st_hprd_total"),
    "hprd_rn_ratio_state": ("hprd_rn", "st_hprd_rn"),
}
STATE_LEVEL = {"state_n_fines": "st_n_fines", "state_fine_amount": "st_fine_amount"}
COLUMNS = [*DIFFS, *RATIOS, *STATE_LEVEL]

FACILITY_COLS = sorted({f for f, _ in (*DIFFS.values(), *RATIOS.values())})
STATE_COLS = sorted({s for _, s in (*DIFFS.values(), *RATIOS.values())} | set(STATE_LEVEL.values()))


def build(reader: FeatureReader, rows: pl.DataFrame, t: str, cfg: FeaturesConfig) -> pl.DataFrame:
    provider = reader.read("provider_by_vintage", t)
    states = reader.read("state_averages_by_vintage", t)
    assert provider is not None and states is not None
    facility = provider.select("ccn", "state", *(pl.col(c).cast(pl.Float64) for c in FACILITY_COLS))
    state = states.filter(pl.col("state") != NATION).select(
        "state", *(pl.col(c).cast(pl.Float64) for c in STATE_COLS)
    )
    joined = facility.join(state, on="state", how="left", validate="m:1")
    out = attach(rows, joined).with_columns(
        *((pl.col(f) - pl.col(s)).alias(name) for name, (f, s) in DIFFS.items()),
        *(
            pl.when(pl.col(s) > 0).then(pl.col(f) / pl.col(s)).alias(name)
            for name, (f, s) in RATIOS.items()
        ),
        *(pl.col(s).alias(name) for name, s in STATE_LEVEL.items()),
    )
    return out.select("ccn", "t", *COLUMNS)
