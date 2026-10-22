"""features_context on the toy world (spec §8.7). Every toy facility is in AL with hprd_total 3.7."""

import polars as pl
import pytest
from silver_fakes import VINTAGES, V

from precursorintelligence.gold.build import build_spine
from precursorintelligence.gold.features import build_features
from precursorintelligence.gold.features.context import COLUMNS


@pytest.fixture(scope="module")
def spine(silver):
    return build_spine(
        silver["facility_universe"], published_vintages=VINTAGES, spine_start="2023-01", v=V
    )


@pytest.fixture(scope="module")
def ctx(silver, spine):
    return build_features("context", silver, spine)


def _row(df, ccn, t):
    return df.filter(pl.col("ccn") == ccn, pl.col("t") == t).row(0, named=True)


def test_one_row_per_spine_row_with_the_spec_columns(ctx, spine):
    assert ctx.select("ccn", "t").equals(spine.select("ccn", "t"))
    assert ctx.columns == ["ccn", "t", *COLUMNS]


@pytest.mark.parametrize(
    ("t", "rate"),
    [
        ("2023-07", 0.0),  # Sunny Acres' G (2023-06-15) is not posted yet
        ("2023-08", 0.2),  # it is now: 1 of 5 facilities
        ("2024-05", 0.25),  # Pine View closed; Maple House's G (2024-02-10) is not posted yet
        ("2024-06", 0.5),  # Maple House's G is posted: 2 of 4
        ("2024-07", 0.25),  # Sunny Acres' G left the 12-month window on 2024-06-15
    ],
)
def test_state_gplus_rate_uses_only_what_was_posted_at_t(ctx, t, rate):
    assert set(ctx.filter(pl.col("t") == t)["state_gplus_rate_12m"]) == {rate}


def test_relative_to_the_state_median(ctx):
    # 2023-08 sev_points_24m: Sunny Acres 28, Oak Grove 12, Pine View 8, Maple House 4, Cedar Hill 4.
    assert _row(ctx, "015001", "2023-08")["rel_sev_points_24m"] == 20.0
    assert _row(ctx, "015004", "2023-08")["rel_sev_points_24m"] == -4.0
    row = _row(ctx, "015002", "2023-08")
    assert (row["state_median_hprd_total"], row["rel_hprd_total"]) == (3.7, 1.0)


def test_eras(ctx):
    assert set(ctx["era"]) == {"pre_2025_07"}  # the toy world ends at 2025-03
    assert set(ctx["penalty_regime_2025_10"]) == {False}


def test_a_facility_without_a_state_gets_null_state_comparisons(silver, spine):
    prov = silver["provider_by_vintage"].with_columns(
        state=pl.when(pl.col("ccn") == "015002").then(None).otherwise(pl.col("state"))
    )
    out = build_features("context", silver | {"provider_by_vintage": prov}, spine)
    row = _row(out, "015002", "2023-08")
    assert row["state_gplus_rate_12m"] is None and row["rel_sev_points_24m"] is None
    assert _row(out, "015001", "2023-08")["state_gplus_rate_12m"] == 0.25  # 1 of the 4 in AL
