"""A month against its schema YAML with Pandera: each planted problem is caught, a clean month passes."""

import polars as pl
import pytest
from validation_fakes import citations

from precursorintelligence.common.config import ConfigError
from precursorintelligence.validation.schema import ColumnRule, TableSchema
from precursorintelligence.validation.validate import ERROR, WARNING, check_schema

SCHEMA = TableSchema(
    table="citations_by_vintage",
    layer="silver",
    columns={
        "ccn": ColumnRule(dtype="String", nullable=False, str_length=6),
        "survey_date": ColumnRule(dtype="Date", nullable=False),
        "scope_severity": ColumnRule(dtype="String", nullable=False, allowed=list("ABCDEFGHIJKL")),
        "inspection_cycle": ColumnRule(dtype="Int64", nullable=True, min=1, max=3),
        "is_complaint": ColumnRule(dtype="Boolean"),
        "vintage_month": ColumnRule(dtype="String", nullable=False),
    },
)


def check(df, schema=SCHEMA):
    return check_schema(df, schema, month="2026-07")


def only(anomalies):
    assert len(anomalies) == 1, anomalies
    (a,) = anomalies
    assert a.severity == ERROR and a.table == "citations_by_vintage" and a.month == "2026-07"
    return a


def test_a_clean_month_passes():
    assert check(citations(n=8)) == []


def test_a_missing_column_is_caught():
    a = only(check(citations().drop("scope_severity")))
    assert (a.check, a.column) == ("missing_column", "scope_severity")


def test_two_missing_columns_are_two_anomalies():
    found = check(citations().drop("scope_severity", "is_complaint"))
    assert sorted(a.column for a in found) == ["is_complaint", "scope_severity"]


def test_an_unexpected_column_is_caught():
    a = only(check(citations().with_columns(pl.lit(1).alias("star_rating"))))
    assert (a.check, a.column) == ("unexpected_column", "star_rating")


def test_a_wrong_dtype_is_caught():
    df = citations().with_columns(pl.col("inspection_cycle").cast(pl.Float64))
    a = only(check(df))
    assert (a.check, a.column, a.observed, a.expected) == (
        "wrong_dtype",
        "inspection_cycle",
        "Float64",
        "Int64",
    )


def test_a_null_in_a_non_nullable_column_is_caught():
    a = only(check(citations(n=4, scope_severity=["D", None, "E", "G"])))
    assert (a.check, a.column, a.observed["rows"]) == ("null_values", "scope_severity", 1)


def test_an_out_of_range_value_is_caught():
    a = only(check(citations(n=4, inspection_cycle=[1, 2, 9, 1])))
    assert (a.check, a.column, a.expected) == ("out_of_range", "inspection_cycle", [1, 3])
    assert a.observed == {"rows": 1, "examples": ["9"]}


def test_a_value_outside_the_allowed_set_is_caught():
    a = only(check(citations(n=4, scope_severity=["D", "Z", "Z", "E"])))
    assert (a.check, a.column) == ("allowed_values", "scope_severity")
    assert a.observed == {"rows": 2, "examples": ["Z"]}


def test_a_short_ccn_is_caught():
    a = only(check(citations(n=4, ccn=["015001", "15002", "015003", "015004"])))
    assert (a.check, a.column, a.expected) == ("str_length", "ccn", 6)


def test_one_sided_ranges_work():
    schema = SCHEMA.model_copy(deep=True)
    schema.columns["inspection_cycle"] = ColumnRule(dtype="Int64", min=2)
    a = only(check(citations(n=4), schema))
    assert a.check == "out_of_range" and a.observed["rows"] == 2  # the two 1s


def test_an_unknown_dtype_in_the_yaml_is_a_config_error():
    schema = SCHEMA.model_copy(deep=True)
    schema.columns["ccn"] = ColumnRule(dtype="Strng")
    with pytest.raises(ConfigError, match="Strng"):
        check(citations(), schema)


def test_a_value_outside_a_warning_only_list_is_a_warning_not_an_error():
    rules = SCHEMA.columns | {
        "scope_severity": ColumnRule(dtype="String", allowed=["D", "E"], allowed_severity="warning")
    }
    found = check(citations(n=4), SCHEMA.model_copy(update={"columns": rules}))  # D, E, G, D
    assert [(a.check, a.severity, a.column) for a in found] == [
        ("allowed_values", WARNING, "scope_severity")
    ]
    assert found[0].observed == {"rows": 1, "examples": ["G"]}


def test_a_value_not_matching_its_pattern_is_caught():
    rules = SCHEMA.columns | {
        "ccn": ColumnRule(dtype="String", nullable=False, pattern=r"^[0-9A-Z]{6}$")
    }
    df = citations(n=4, ccn=["015001", "01-19A", "015003", "015004"])  # 6 characters, wrong format
    a = only(check(df, SCHEMA.model_copy(update={"columns": rules})))
    assert (a.check, a.column, a.expected) == ("pattern", "ccn", r"^[0-9A-Z]{6}$")
    assert a.observed == {"rows": 1, "examples": ["01-19A"]}


def test_a_duplicate_key_is_caught():
    schema = SCHEMA.model_copy(update={"unique": ["ccn", "survey_date"]})
    df = citations(n=4)
    df = pl.concat([df, df.head(1)])
    a = only(check(df, schema))
    assert (a.check, a.column, a.expected) == (
        "duplicate_key",
        "ccn,survey_date",
        ["ccn", "survey_date"],
    )
    assert a.observed == {"rows": 2, "examples": ["015000|2026-05-01"]}


def test_rows_sharing_part_of_the_key_are_not_duplicates():
    schema = SCHEMA.model_copy(update={"unique": ["ccn", "survey_date"]})
    assert check(citations(n=4, ccn=["015001"] * 4), schema) == []  # four survey dates


def test_an_empty_month_is_an_error():
    a = only(check(citations(n=4).clear()))
    assert (a.check, a.column, a.observed) == ("empty_month", None, 0)
