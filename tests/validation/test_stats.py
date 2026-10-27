"""Per-month column statistics."""

import json

import polars as pl
import pytest
from validation_fakes import citations

from precursorintelligence.validation.stats import table_stats


def stats(df, top_values=5):
    return table_stats(df, table="citations_by_vintage", month="2026-07", top_values=top_values)


def test_row_count_and_one_entry_per_column():
    s = stats(citations(n=8))
    assert s["table"] == "citations_by_vintage" and s["month"] == "2026-07"
    assert s["row_count"] == 8
    assert list(s["columns"]) == citations().columns


def test_numbers_get_min_max_mean_std():
    col = stats(citations(n=4))["columns"]["inspection_cycle"]  # 1, 2, 3, 1
    assert (col["min"], col["max"], col["distinct"]) == (1, 3, 3)
    assert col["mean"] == pytest.approx(1.75)
    assert col["std"] == pytest.approx(pl.Series([1, 2, 3, 1]).std())


def test_null_pct_counts_nulls_over_all_rows():
    df = citations(n=4, inspection_cycle=[1, None, None, 2])
    col = stats(df)["columns"]["inspection_cycle"]
    assert col["null_pct"] == 50.0 and col["mean"] == pytest.approx(1.5)


def test_text_gets_the_most_frequent_values_first():
    col = stats(citations(n=4), top_values=2)["columns"]["scope_severity"]  # D, E, G, D
    assert col["top_values"][0] == ["D", 2] and len(col["top_values"]) == 2
    assert "mean" not in col


def test_dates_get_min_and_max_as_iso_text():
    col = stats(citations(n=4))["columns"]["survey_date"]
    assert (col["min"], col["max"]) == ("2026-05-01", "2026-05-04")


def test_an_empty_month_has_no_numbers_but_still_serialises():
    s = stats(citations(n=4).clear())
    assert s["row_count"] == 0 and s["columns"]["inspection_cycle"]["mean"] is None
    json.dumps(s)


def test_the_stats_are_json():
    json.dumps(stats(citations(n=8)))
