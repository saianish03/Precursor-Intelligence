"""features_quality on the toy world (spec §8.6). The toy world has 3 of the 16 MDS measures (410, the bridge
419 -> 481 at 2024-06, 434) and 2 claims measures (521, 551); the rest are always missing."""

import polars as pl
import pytest
from silver_fakes import VINTAGES, V

from precursorintelligence.gold import GoldCheckError
from precursorintelligence.gold.build import build_spine
from precursorintelligence.gold.features import build_features
from precursorintelligence.gold.features.quality import DAYS_PER_MONTH, columns, qm_config

MDS = "mds_qm_by_vintage"


@pytest.fixture(scope="module")
def spine(silver):
    return build_spine(
        silver["facility_universe"], published_vintages=VINTAGES, spine_start="2023-01", v=V
    )


@pytest.fixture(scope="module")
def qual(silver, spine):
    return build_features("quality", silver, spine)


def _row(df, ccn, t):
    return df.filter(pl.col("ccn") == ccn, pl.col("t") == t).row(0, named=True)


def _with_mds(silver, mds):
    return silver | {MDS: mds}


def test_one_row_per_spine_row_with_the_contract_columns(qual, spine):
    assert qual.select("ccn", "t").equals(spine.select("ccn", "t"))
    assert qual.columns == ["ccn", "t", *columns(qm_config())]
    assert len(qual.columns) - 2 == 41  # 16 measures x 2, 4 deltas, missing count, 2 ages, 2 flags


def test_a_bridge_is_one_feature_across_the_code_change(qual):
    old, new = _row(qual, "015001", "2024-05"), _row(qual, "015001", "2024-06")
    assert old["qm_ls_antipsychotic"] is not None and new["qm_ls_antipsychotic"] is not None


def test_both_codes_of_a_bridge_in_one_vintage_stop_the_build(silver, spine):
    mds = silver[MDS]
    twin = mds.filter(pl.col("measure_code") == "419", pl.col("vintage_month") == "2024-05").head(1)
    with pytest.raises(GoldCheckError, match="repeat a"):
        build_features(
            "quality",
            _with_mds(silver, pl.concat([mds, twin.with_columns(measure_code=pl.lit("481"))])),
            spine,
        )


def test_an_unrefreshed_period_gives_a_null_delta(silver, spine, qual):
    assert _row(qual, "015001", "2024-01")["d6_qm_ls_falls_major_injury"] is not None
    frozen = silver[MDS].with_columns(
        measure_period_end=pl.when(pl.col("measure_code") == "410")
        .then(pl.col("measure_period_end").min().over("measure_code"))
        .otherwise(pl.col("measure_period_end"))
    )
    got = build_features("quality", _with_mds(silver, frozen), spine)
    assert got["d6_qm_ls_falls_major_injury"].null_count() == got.height


def test_a_facility_missing_from_the_tables_has_no_values_and_footnote_none(silver, spine, qual):
    assert _row(qual, "015001", "2024-01")["qm_missing_count"] == 11  # 5 toy measures of 16
    gone = {
        table: silver[table].filter(
            ~((pl.col("ccn") == "015001") & (pl.col("vintage_month") == "2024-01"))
        )
        for table in (MDS, "claims_qm_by_vintage")
    }
    row = _row(build_features("quality", silver | gone, spine), "015001", "2024-01")
    assert row["qm_missing_count"] == 16
    assert row["qm_ss_rehosp"] is None and row["qm_ss_rehosp_fn"] == "none"
    assert row["qm_age_months_mds"] is None and row["d6_qm_ls_falls_major_injury"] is None


def test_qm_age_counts_months_from_the_period_end_to_the_cutoff(silver, spine, qual):
    end = (
        silver[MDS]
        .filter(pl.col("ccn") == "015001", pl.col("vintage_month") == "2024-08")[
            "measure_period_end"
        ]
        .max()
    )
    cutoff = spine.filter(pl.col("ccn") == "015001", pl.col("t") == "2024-08")["cutoff_date"][0]
    expected = (cutoff - end).days / DAYS_PER_MONTH
    assert _row(qual, "015001", "2024-08")["qm_age_months_mds"] == pytest.approx(expected)


def test_definition_flags_are_true_from_their_month_on(qual):
    for flag, month in qm_config().definition_changes.items():
        assert qual[flag].to_list() == (qual["t"] >= month).to_list(), flag


def test_only_the_selected_measures_footnotes_are_checked(silver, spine):
    mds = silver[MDS]
    dropped = mds.head(1).with_columns(measure_code=pl.lit("415"), four_quarter_avg_fn=pl.lit("99"))
    build_features("quality", _with_mds(silver, pl.concat([mds, dropped])), spine)  # 415 is dropped
    selected = mds.with_columns(four_quarter_avg_fn=pl.lit("99"))
    with pytest.raises(GoldCheckError, match="99"):
        build_features("quality", _with_mds(silver, selected), spine)


def test_footnote_21_is_a_measure_that_cms_could_not_validate(silver, spine):
    # Real data: 419 / 481 carry 21 without a value in 2023-03..2026-10 (7,034 rows on the ml-dev silver).
    mds = silver[MDS].with_columns(
        four_quarter_avg=pl.when(pl.col("measure_code") == "419")
        .then(None)
        .otherwise(pl.col("four_quarter_avg")),
        four_quarter_avg_fn=pl.when(pl.col("measure_code") == "419").then(pl.lit("21")),
    )
    row = _row(build_features("quality", _with_mds(silver, mds), spine), "015001", "2024-05")
    assert row["qm_ls_antipsychotic"] is None
    assert row["qm_ls_antipsychotic_fn"] == "measure_not_validated"
