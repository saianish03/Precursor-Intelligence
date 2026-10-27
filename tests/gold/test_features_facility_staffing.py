"""features_facility_staffing on the toy world (spec §8.5). Provider values are constant over time in the
toy world (silver_fakes.py), so a Δ is 0 whenever t−k exists, and null when it was never published."""

import polars as pl
import pytest
from silver_fakes import VINTAGES, V

from precursorintelligence.gold import GoldCheckError
from precursorintelligence.gold.build import build_spine
from precursorintelligence.gold.features import build_features
from precursorintelligence.gold.features.facility_staffing import COLUMNS


@pytest.fixture(scope="module")
def spine(silver):
    return build_spine(
        silver["facility_universe"], published_vintages=VINTAGES, spine_start="2023-01", v=V
    )


@pytest.fixture(scope="module")
def staff(silver, spine):
    return build_features("facility_staffing", silver, spine)


def _row(df, ccn, t):
    return df.filter(pl.col("ccn") == ccn, pl.col("t") == t).row(0, named=True)


def test_one_row_per_spine_row_with_the_spec_columns(staff, spine):
    assert staff.select("ccn", "t").equals(spine.select("ccn", "t"))
    assert staff.columns == ["ccn", "t", *COLUMNS]
    assert len(COLUMNS) == 47  # the facility_staffing columns of the column contract


def test_derived_values(staff):
    row = _row(
        staff, "015001", "2024-01"
    )  # 120 beds, 96 residents a day, first approved 1990-01-15
    assert row["occupancy"] == pytest.approx(0.8)
    assert row["rn_share"] == pytest.approx(0.6 / 3.7)
    assert row["weekend_gap"] == pytest.approx(3.7 - 3.3)
    assert row["facility_age_years"] == pytest.approx(12404 / 365.25)
    assert row["sff_status"] == "NONE"
    assert (row["state"], row["ownership_group"], row["chain_id"]) == ("AL", "for_profit", "900001")


def test_chain_id_is_null_before_2023_06(staff):
    assert _row(staff, "015001", "2023-05")["chain_id"] is None


def test_footnotes_become_reasons_and_none(silver, spine, staff):
    assert _row(staff, "015001", "2024-01")["turnover_total_fn"] == "none"
    prov = silver["provider_by_vintage"].with_columns(
        turnover_total_fn=pl.when(pl.col("ccn") == "015002").then(pl.lit("27")),
        staffing_rating_fn=pl.when(pl.col("ccn") == "015002").then(pl.lit("12")),
    )
    row = _row(
        build_features("facility_staffing", silver | {"provider_by_vintage": prov}, spine),
        "015002",
        "2024-01",
    )
    assert row["turnover_total_fn"] == "turnover_criteria_not_met"
    assert row["staffing_rating_fn"] == "staffing_rating_downgraded"


def test_an_unmapped_footnote_code_stops_the_build(silver, spine):
    prov = silver["provider_by_vintage"].with_columns(pt_staffing_fn=pl.lit("99"))
    with pytest.raises(GoldCheckError, match="99"):
        build_features("facility_staffing", silver | {"provider_by_vintage": prov}, spine)


def test_occupancy_is_capped_and_null_without_beds(silver, spine):
    prov = silver["provider_by_vintage"].with_columns(
        avg_residents_per_day=pl.when(pl.col("ccn") == "015002")
        .then(500.0)
        .otherwise(pl.col("avg_residents_per_day")),
        certified_beds=pl.when(pl.col("ccn") == "015004")
        .then(None)
        .otherwise(pl.col("certified_beds")),
    )
    out = build_features("facility_staffing", silver | {"provider_by_vintage": prov}, spine)
    assert _row(out, "015002", "2024-01")["occupancy"] == 1.5
    assert _row(out, "015004", "2024-01")["occupancy"] is None


@pytest.mark.parametrize(
    ("t", "d6", "d12"),
    [
        ("2023-03", None, None),  # no history before 2023-01 in the toy world
        ("2023-07", 0.0, None),  # t−6 = 2023-01 exists, t−12 = 2022-07 does not
        ("2024-01", 0.0, 0.0),
        ("2024-06", None, 0.0),  # t−6 = 2023-12 was never published
        ("2024-12", 0.0, None),  # t−12 = 2023-12 was never published
    ],
)
def test_deltas_are_null_when_t_minus_k_was_never_published(staff, t, d6, d12):
    row = _row(staff, "015001", t)
    assert (row["d6_hprd_total"], row["d6_hprd_rn"]) == (d6, d6)
    assert (row["d12_hprd_total"], row["d12_turnover_total"]) == (d12, d12)


def test_deltas_are_null_when_the_facility_was_absent_at_t_minus_k(staff):
    # Cedar Hill first reports in 2023-04, so at 2023-07 its t−6 (2023-01) has no row.
    assert _row(staff, "015005", "2023-07")["d6_hprd_total"] is None
    assert _row(staff, "015001", "2023-07")["d6_hprd_total"] == 0.0


def test_deltas_measure_the_change(silver, spine):
    prov = silver["provider_by_vintage"].with_columns(
        hprd_total=pl.when(pl.col("vintage_month") >= "2024-01").then(4.0).otherwise(3.7)
    )
    row = _row(
        build_features("facility_staffing", silver | {"provider_by_vintage": prov}, spine),
        "015001",
        "2024-01",
    )
    assert row["d6_hprd_total"] == pytest.approx(0.3)
    assert row["d12_hprd_total"] == pytest.approx(0.3)
