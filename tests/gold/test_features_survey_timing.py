"""features_survey_timing on the toy world (spec §8.2). Expected values are worked out from the story in
silver_fakes.py: standard surveys and cited complaint surveys, each visible from the report that posts it."""

from datetime import date

import polars as pl
import pytest
from silver_fakes import cutoff

from precursorintelligence.gold.build import build_spine
from precursorintelligence.gold.features import build_features
from precursorintelligence.gold.features.survey_timing import COLUMNS


@pytest.fixture(scope="module")
def spine(silver):
    from silver_fakes import VINTAGES, V

    return build_spine(
        silver["facility_universe"], published_vintages=VINTAGES, spine_start="2023-01", v=V
    )


@pytest.fixture(scope="module")
def timing(silver, spine):
    return build_features("survey_timing", silver, spine)


def _row(df, ccn, t):
    return df.filter(pl.col("ccn") == ccn, pl.col("t") == t).row(0, named=True)


def _days(t, d):
    return float((cutoff(t) - d).days)


def test_one_row_per_spine_row_with_the_spec_columns(timing, spine):
    assert timing.select("ccn", "t").equals(spine.select("ccn", "t"))
    assert timing.columns == ["ccn", "t", *COLUMNS]


def test_a_standard_survey_counts_only_once_it_is_posted(timing):
    # Sunny Acres' 2023-06-15 standard survey is first posted in 2023-08.
    before = _row(timing, "015001", "2023-07")
    assert before["days_since_last_std"] == _days("2023-07", date(2022, 5, 10))
    assert before["n_std_36m"] == 1
    after = _row(timing, "015001", "2023-08")
    assert after["days_since_last_std"] == _days("2023-08", date(2023, 6, 15))
    assert after["n_std_36m"] == 2


def test_due_and_overdue_follow_the_456_day_rule(timing):
    on_time = _row(timing, "015001", "2023-08")
    assert on_time["days_to_std_due"] == 456 - on_time["days_since_last_std"]
    assert on_time["std_overdue"] is False
    late = _row(timing, "015001", "2024-10")  # 474 days after 2023-06-15
    assert late["days_since_last_std"] == 474.0
    assert late["days_to_std_due"] == -18.0
    assert late["std_overdue"] is True


def test_a_late_posted_complaint_survey_is_invisible_before_its_report(timing):
    # Maple House's cited complaint survey (2024-02-10) is posted in 2024-06.
    may = _row(timing, "015004", "2024-05")
    assert may["n_complaint_surveys_cited_12m"] == 0
    assert may["days_since_last_complaint_survey_cited"] is None
    june = _row(timing, "015004", "2024-06")
    assert june["n_complaint_surveys_cited_12m"] == 1
    assert june["n_complaint_surveys_cited_24m"] == 1
    assert june["days_since_last_complaint_survey_cited"] == _days("2024-06", date(2024, 2, 10))


def test_windows_look_back_from_the_cutoff(timing):
    # Oak Grove's complaint survey 2024-04-18 leaves the 12-month window after 2025-04; it stays in 24m.
    row = _row(timing, "015002", "2025-03")
    assert (row["n_complaint_surveys_cited_12m"], row["n_complaint_surveys_cited_24m"]) == (1, 1)
    assert row["n_std_36m"] == 2  # 2022-09-14 and 2023-10-05, both after 2022-03-01
    assert row["n_ic_surveys_12m"] == 0  # no infection-control surveys in this world


def test_fire_surveys_and_surveys_after_the_cutoff_are_ignored(silver, spine):
    extra = pl.DataFrame(
        {
            "ccn": ["015002", "015002"],
            "survey_date": [
                date(2024, 8, 1),
                date(2024, 9, 5),
            ],  # fire; and 4 days after the cutoff
            "survey_type": ["FIRE_STANDARD", "HEALTH_STANDARD"],
            "survey_cycle": [1, 1],
            "survey_fn": [None, None],
            "vintage_month": ["2024-09", "2024-09"],
            "cutoff_date": [cutoff("2024-09")] * 2,
        },
        schema=silver["surveys_by_vintage"].schema,
    )
    world = silver | {"surveys_by_vintage": pl.concat([silver["surveys_by_vintage"], extra])}
    row = _row(build_features("survey_timing", world, spine), "015002", "2024-09")
    assert row["days_since_last_std"] == _days("2024-09", date(2023, 10, 5))


def test_a_facility_without_a_standard_survey(silver, spine):
    no_std = silver["surveys_by_vintage"].filter(
        ~((pl.col("ccn") == "015004") & (pl.col("survey_type") == "HEALTH_STANDARD"))
    )
    row = _row(
        build_features("survey_timing", silver | {"surveys_by_vintage": no_std}, spine),
        "015004",
        "2024-06",
    )
    assert row["no_std_on_record"] is True
    assert row["days_since_last_std"] is None
    assert row["days_to_std_due"] is None
    assert row["std_overdue"] is None
    assert row["n_std_36m"] == 0
