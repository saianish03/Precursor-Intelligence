"""``features_covid`` (docs/ALL_TABLES_CONTRACT.md): COVID-19 vaccination of residents and staff, from
``covid_vax_by_vintage`` at t.

CMS published the file only 2021-09 → 2025-07, so the read is optional: a month without it gets
``covid_available`` False and null rates. A facility missing from a published file, or whose
``vax_data_date`` is after the cutoff, gets null rates with ``covid_available`` True.
"""

from __future__ import annotations

import polars as pl

from ...common.config import FeaturesConfig
from .common import CUTOFF, attach
from .reader import FeatureReader

COLUMNS = ["covid_available", "covid_pct_residents_vax", "covid_pct_staff_vax"]


def build(reader: FeatureReader, rows: pl.DataFrame, t: str, cfg: FeaturesConfig) -> pl.DataFrame:
    vax = reader.read("covid_vax_by_vintage", t, optional=True)
    if vax is None:
        return rows.select(
            "ccn",
            "t",
            covid_available=pl.lit(False),
            covid_pct_residents_vax=pl.lit(None, pl.Float64),
            covid_pct_staff_vax=pl.lit(None, pl.Float64),
        )
    per_ccn = vax.filter(
        pl.col("vax_data_date").is_null() | (pl.col("vax_data_date") <= CUTOFF)
    ).select(
        "ccn",
        covid_pct_residents_vax=pl.col("pct_residents_vax").cast(pl.Float64),
        covid_pct_staff_vax=pl.col("pct_staff_vax").cast(pl.Float64),
    )
    return (
        attach(rows, per_ccn)
        .with_columns(covid_available=pl.lit(True))
        .select("ccn", "t", *COLUMNS)
    )
