"""Rows belong to the month they are stored under: vintage_month, cutoff_date and gold's first month."""

from datetime import date

import polars as pl
from validation_fakes import citations

from precursorintelligence.common.config import load_splits_config
from precursorintelligence.ingestion.pipeline import EXIT_VALIDATION
from precursorintelligence.validation.run import run_validation, write_schemas
from precursorintelligence.validation.validate import ERROR, check_month

SPINE_START = load_splits_config().spine_start


def found(df, month="2026-07", first_month=None):
    return [
        (a.check, a.column, a.severity)
        for a in check_month(df, table="t", month=month, first_month=first_month)
    ]


def test_a_month_whose_rows_carry_its_own_month_passes():
    df = citations(n=4).with_columns(pl.lit(date(2026, 7, 1)).alias("cutoff_date"))
    assert found(df) == []


def test_a_vintage_month_that_differs_from_the_partition_is_an_error():
    df = citations(n=4, vintage_month=["2026-07", "2026-07", "2026-06", "2026-07"])
    (a,) = check_month(df, table="t", month="2026-07")
    assert (a.check, a.column, a.severity, a.expected) == (
        "month_mismatch",
        "vintage_month",
        ERROR,
        "2026-07",
    )
    assert a.observed == {"rows": 1, "examples": ["2026-06"]}


def test_a_cutoff_date_that_is_not_the_first_day_of_the_month_is_an_error():
    df = citations(n=2).with_columns(pl.Series("cutoff_date", [date(2026, 7, 1), date(2026, 6, 1)]))
    assert found(df) == [("month_mismatch", "cutoff_date", ERROR)]


def test_a_month_before_the_first_month_is_an_error():
    df = pl.DataFrame({"ccn": ["015001"], "t": ["2022-12"], "cutoff_date": [date(2022, 12, 1)]})
    assert found(df, month="2022-12", first_month="2023-01") == [
        ("before_first_month", None, ERROR)
    ]
    assert found(df.with_columns(t=pl.lit("2023-01"), cutoff_date=pl.lit(date(2023, 1, 1))),
                 month="2023-01", first_month="2023-01") == []  # fmt: skip


def test_a_mislabelled_silver_partition_fails_the_run(put_silver, ctx, cfg):
    put_silver("citations_by_vintage", "2026-07", citations(n=8, vintage="2026-07"))
    write_schemas(ctx, cfg, layer="silver")
    put_silver(
        "citations_by_vintage", "2026-08", citations(n=8, vintage="2026-07")
    )  # copied, not rebuilt
    res = run_validation(ctx, cfg, layer="silver")
    assert res.exit_code == EXIT_VALIDATION
    assert [(a.month, a.check, a.column) for a in res.anomalies] == [
        ("2026-08", "month_mismatch", "vintage_month")
    ]


def spine(*months, cutoff=None):
    rows = [
        {"ccn": f"0150{i:02d}", "t": t, "cutoff_date": cutoff or date(int(t[:4]), int(t[5:]), 1)}
        for t in months
        for i in range(20)
    ]
    return pl.DataFrame(rows)


def test_gold_checks_cutoff_date_against_t_and_the_spine_start(put_gold, ctx, cfg):
    put_gold("spine", spine(SPINE_START))
    write_schemas(ctx, cfg, layer="gold")
    before = f"{int(SPINE_START[:4]) - 1}-12"
    put_gold("spine", pl.concat([spine(before), spine(SPINE_START, cutoff=date(2020, 1, 1))]))
    res = run_validation(ctx, cfg, layer="gold", all_months=True)
    assert [(a.month, a.check, a.column) for a in res.anomalies] == [
        (before, "before_first_month", None),
        (SPINE_START, "month_mismatch", "cutoff_date"),
    ]


def test_a_month_that_is_not_yyyy_mm_is_one_error_not_a_crash():
    df = pl.DataFrame({"ccn": ["015001"], "t": ["2026-13"], "cutoff_date": [date(2026, 1, 1)]})
    for month in ("2026-13", "junk", "2026-1"):
        assert found(df, month=month, first_month="2023-01") == [("bad_month", None, ERROR)]


def test_a_malformed_gold_month_fails_the_run_with_exit_2(put_gold, ctx, cfg):
    put_gold("spine", spine(SPINE_START))
    write_schemas(ctx, cfg, layer="gold")
    bad = spine(SPINE_START).with_columns(t=pl.lit("2026-13"))
    put_gold("spine", pl.concat([spine(SPINE_START), bad]))
    res = run_validation(ctx, cfg, layer="gold", all_months=True)
    assert res.exit_code == EXIT_VALIDATION
    assert ("2026-13", "bad_month") in {(a.month, a.check) for a in res.anomalies}
