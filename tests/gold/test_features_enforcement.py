"""features_enforcement on the toy world (spec §8.4): Sunny Acres' $15,000 fine (2023-06-15, posted 2023-09)
and Oak Grove's two identical $5,000 fines (2023-10-05, posted 2024-01, dup_seq 1 and 2)."""

import math
from datetime import date

import polars as pl
import pytest
from silver_fakes import VINTAGES, V, cutoff

from precursorintelligence.gold.build import build_spine
from precursorintelligence.gold.features import build_features
from precursorintelligence.gold.features.enforcement import COLUMNS


@pytest.fixture(scope="module")
def spine(silver):
    return build_spine(
        silver["facility_universe"], published_vintages=VINTAGES, spine_start="2023-01", v=V
    )


@pytest.fixture(scope="module")
def enf(silver, spine):
    return build_features("enforcement", silver, spine)


def _row(df, ccn, t):
    return df.filter(pl.col("ccn") == ccn, pl.col("t") == t).row(0, named=True)


def test_one_row_per_spine_row_with_the_spec_columns(enf, spine):
    assert enf.select("ccn", "t").equals(spine.select("ccn", "t"))
    assert enf.columns == ["ccn", "t", *COLUMNS]


def test_a_fine_counts_from_the_report_that_posts_it(enf):
    aug, sep = _row(enf, "015001", "2023-08"), _row(enf, "015001", "2023-09")
    assert (aug["n_fines_24m"], aug["fine_dollars_24m"], aug["any_penalty_36m"]) == (0, 0.0, False)
    assert aug["days_since_last_penalty"] is None
    assert (sep["n_fines_24m"], sep["max_fine_24m"], sep["any_penalty_36m"]) == (1, 15000.0, True)
    assert sep["fine_dollars_24m"] == pytest.approx(math.log1p(15000.0))
    assert sep["days_since_last_penalty"] == float((cutoff("2023-09") - date(2023, 6, 15)).days)


def test_identical_fines_both_count(enf):
    row = _row(enf, "015002", "2024-01")
    assert (row["n_fines_24m"], row["max_fine_24m"]) == (2, 5000.0)
    assert row["fine_dollars_24m"] == pytest.approx(math.log1p(10000.0))


def test_fines_leave_the_24_month_window(enf):
    # 2023-06-15 is within (cutoff - 24 months, cutoff] up to t = 2025-06; still in at V = 2025-03.
    assert _row(enf, "015001", V)["n_fines_24m"] == 1


def test_penalties_after_the_cutoff_are_ignored(silver, spine):
    # Silver keeps a few events dated just after their file's cutoff (SILVER_CONTRACT §9); a feature must not.
    late = pl.DataFrame(
        {
            "ccn": ["015004", "015004"], "penalty_date": [date(2024, 6, 5), date(2024, 6, 9)],
            "penalty_type": ["Fine", "Payment Denial"], "fine_amount": [9000.0, None],
            "payment_denial_start_date": [None, date(2024, 6, 9)],
            "payment_denial_length_days": [None, 20.0], "fine_id": [None, None], "dup_seq": [1, 1],
            "penalty_event_key": ["015004|2024-06-05|late", "015004|2024-06-09|late"],
            "vintage_month": ["2024-06"] * 2, "cutoff_date": [cutoff("2024-06")] * 2,
        },
        schema=silver["penalties_by_vintage"].schema,
    )  # fmt: skip
    world = silver | {"penalties_by_vintage": pl.concat([silver["penalties_by_vintage"], late])}
    row = _row(build_features("enforcement", world, spine), "015004", "2024-06")
    assert row == _row(build_features("enforcement", silver, spine), "015004", "2024-06")
    assert (row["n_fines_24m"], row["n_denials_24m"], row["any_penalty_36m"]) == (0, 0, False)
    assert row["days_since_last_penalty"] is None


def test_payment_denials(silver, spine):
    denial = pl.DataFrame(
        {
            "ccn": ["015004"], "penalty_date": [date(2024, 3, 1)],
            "penalty_type": ["Payment Denial"], "fine_amount": [None],
            "payment_denial_start_date": [date(2024, 3, 15)], "payment_denial_length_days": [30.0],
            "fine_id": [None], "dup_seq": [1], "penalty_event_key": ["015004|2024-03-01|denial"],
            "vintage_month": ["2024-06"], "cutoff_date": [cutoff("2024-06")],
        },
        schema=silver["penalties_by_vintage"].schema,
    )  # fmt: skip
    world = silver | {"penalties_by_vintage": pl.concat([silver["penalties_by_vintage"], denial])}
    row = _row(build_features("enforcement", world, spine), "015004", "2024-06")
    assert (row["n_denials_24m"], row["denial_days_24m"], row["n_fines_24m"]) == (1, 30.0, 0)
    assert row["any_penalty_36m"] is True
