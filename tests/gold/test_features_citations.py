"""features_citations on the toy world (spec §8.3, §8.1). Expected values come from the citation story in
silver_fakes.py; severity points D 4, E 8, G 20; ordinal D 4, E 5, G 7."""

from datetime import date

import polars as pl
import pytest
from silver_fakes import VINTAGES, V, cutoff

from precursorintelligence.gold.build import build_spine
from precursorintelligence.gold.features import build_features
from precursorintelligence.gold.features.citations import COLUMNS


@pytest.fixture(scope="module")
def spine(silver):
    return build_spine(
        silver["facility_universe"], published_vintages=VINTAGES, spine_start="2023-01", v=V
    )


@pytest.fixture(scope="module")
def cit(silver, spine):
    return build_features("citations", silver, spine)


def _row(df, ccn, t):
    return df.filter(pl.col("ccn") == ccn, pl.col("t") == t).row(0, named=True)


def test_one_row_per_spine_row_with_the_spec_columns(cit, spine):
    assert cit.select("ccn", "t").equals(spine.select("ccn", "t"))
    assert cit.columns == ["ccn", "t", *COLUMNS]


def test_counts_and_points_before_and_after_the_g_is_posted(cit):
    # Sunny Acres: D 2022-05-10, then G + D on 2023-06-15, first posted in 2023-08.
    july = _row(cit, "015001", "2023-07")
    assert (july["n_cit_12m"], july["n_cit_24m"], july["sev_points_24m"]) == (0, 1, 4.0)
    assert july["n_cit_GI_24m"] == 0 and july["days_since_last_gplus"] is None
    aug = _row(cit, "015001", "2023-08")
    assert (aug["n_cit_12m"], aug["n_cit_24m"]) == (2, 3)
    assert (aug["sev_points_12m"], aug["sev_points_24m"]) == (24.0, 28.0)
    assert (aug["n_cit_AC_24m"], aug["n_cit_DF_24m"], aug["n_cit_GI_24m"], aug["n_cit_JL_24m"]) == (
        0, 2, 1, 0,
    )  # fmt: skip
    assert aug["sev_points_per_bed_24m"] == pytest.approx(28.0 / 120.0)
    assert aug["n_gplus_surveys_36m"] == 1
    assert aug["days_since_last_gplus"] == float((cutoff("2023-08") - date(2023, 6, 15)).days)


def test_last_and_previous_standard_survey(cit):
    only_one = _row(cit, "015001", "2023-07")  # one standard survey on record
    assert (only_one["max_sev_last_std"], only_one["gplus_last_std"]) == (4.0, False)
    assert only_one["max_sev_prev_std"] is None and only_one["gplus_prev_std"] is None
    assert only_one["n_repeat_tags_last_std"] is None
    two = _row(cit, "015001", "2023-08")
    assert (two["max_sev_last_std"], two["gplus_last_std"]) == (7.0, True)
    assert (two["max_sev_prev_std"], two["gplus_prev_std"]) == (4.0, False)
    assert two["n_repeat_tags_last_std"] == 1.0  # F0880 on both surveys


def test_no_repeat_tags_when_the_tags_differ(cit):
    # Oak Grove: F0812 + F0880 on 2022-09-14, F0689 + F0761 on 2023-10-05.
    row = _row(cit, "015002", "2023-11")
    assert (row["max_sev_last_std"], row["max_sev_prev_std"]) == (5.0, 5.0)
    assert row["n_repeat_tags_last_std"] == 0.0


def test_a_complaint_g_counts_as_complaint_but_not_on_the_standard_survey(cit):
    # Maple House's complaint G (2024-02-10) is posted in 2024-06.
    may, june = _row(cit, "015004", "2024-05"), _row(cit, "015004", "2024-06")
    assert (may["n_complaint_cit_12m"], may["n_complaint_gplus_24m"]) == (0, 0)
    assert (june["n_complaint_cit_12m"], june["n_complaint_gplus_24m"]) == (1, 1)
    assert june["max_sev_last_std"] == 4.0  # its last standard survey (2022-12-07) had a D only
    assert june["n_gplus_surveys_36m"] == 1


def test_category_counts(cit):
    aug = _row(cit, "015001", "2023-08")
    assert aug["n_ic_cit_24m"] == 2  # F0880 twice, category Infection Control
    assert aug["n_cat_qol_care_24m"] == 1  # F0689
    assert aug["n_cat_abuse_24m"] == 0
    oak = _row(cit, "015002", "2024-06")
    assert oak["n_cat_abuse_24m"] == 1  # F0600 on 2024-04-18


def test_a_facility_with_no_citations_gets_zeros_and_nulls(silver, spine):
    none = silver["citations_by_vintage"].filter(pl.col("ccn") != "015005")
    row = _row(
        build_features("citations", silver | {"citations_by_vintage": none}, spine),
        "015005",
        "2024-08",
    )
    assert row["n_cit_24m"] == 0 and row["sev_points_24m"] == 0.0
    assert row["days_since_last_gplus"] is None
    # its standard surveys are still on record, with no standard citation on them: 0, not null
    assert (row["max_sev_last_std"], row["gplus_last_std"]) == (0.0, False)
    assert row["n_repeat_tags_last_std"] is None


def test_uncorrected_counts_the_configured_values_only(silver, spine):
    cits = silver["citations_by_vintage"].with_columns(
        deficiency_corrected=pl.when(pl.col("ftag") == "F0689")
        .then(pl.lit("Deficient, Provider has plan of correction"))
        .otherwise(pl.col("deficiency_corrected"))
    )
    out = build_features("citations", silver | {"citations_by_vintage": cits}, spine)
    assert _row(out, "015001", "2023-08")["n_uncorrected"] == 1
    assert _row(out, "015005", "2024-08")["n_uncorrected"] == 0  # "date of correction" is corrected


def test_citations_after_the_cutoff_are_ignored(silver, spine):
    late = silver["citations_by_vintage"].filter(pl.col("vintage_month") == "2024-09").head(1)
    late = late.with_columns(survey_date=pl.lit(date(2024, 9, 6)), scope_severity=pl.lit("L"))
    world = silver | {"citations_by_vintage": pl.concat([silver["citations_by_vintage"], late])}
    ccn = late["ccn"][0]
    before = _row(build_features("citations", silver, spine), ccn, "2024-09")
    after = _row(build_features("citations", world, spine), ccn, "2024-09")
    assert after == before
