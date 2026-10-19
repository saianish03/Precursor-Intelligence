"""Type and value normalisation from all-text bronze to typed silver (docs/DATA_PIPELINE_SPEC.md §3.4).

Each dtype of configs/column_map.yaml has one rule. Empty strings and nulls are both "missing" (bronze keeps
the difference, silver does not). A non-empty value a rule cannot convert becomes null and is counted, so the
Checkpoint 2 report shows it; only a bad CCN fails the partition.
"""

from __future__ import annotations

from typing import Any

import polars as pl

CCN_PATTERN = r"^[0-9A-Z]{6}$"
TAG_PATTERN = r"^[0-9]{4}$"
SEVERITIES = list("ABCDEFGHIJKL")
MDS_PERIOD = r"^(\d{4})Q([1-4])-(\d{4})Q([1-4])$"
CLAIMS_PERIOD = r"^(\d{8})-(\d{8})$"

POLARS_TYPES: dict[str, pl.DataType] = {
    "ccn": pl.String(),
    "str": pl.String(),
    "footnote": pl.String(),
    "tag": pl.String(),
    "severity": pl.String(),
    "enum": pl.String(),
    "survey_type_enum": pl.String(),
    "int": pl.Int64(),
    "float": pl.Float64(),
    "date": pl.Date(),
    "bool_yn": pl.Boolean(),
}


class NormalizeError(ValueError):
    """A value that must never be wrong is (a malformed CCN). A data failure: the partition is not written."""


def text(name: str) -> pl.Expr:
    """The trimmed cell, with empty strings as null."""
    s = pl.col(name).str.strip_chars()
    return pl.when(s == "").then(None).otherwise(s)


def _rule(name: str, dtype: str, value_maps: dict[str, Any]) -> pl.Expr:
    t = text(name)
    if dtype == "ccn":
        return t.str.to_uppercase()
    if dtype in ("footnote",):
        return t
    if dtype == "str":
        raw = pl.col(name)
        return pl.when(raw.str.strip_chars() == "").then(None).otherwise(raw)
    if dtype == "tag":
        return pl.when(t.str.contains(TAG_PATTERN)).then(t)
    if dtype == "severity":
        u = t.str.to_uppercase()
        return pl.when(u.is_in(SEVERITIES)).then(u)
    if dtype == "int":
        return t.cast(pl.Int64, strict=False)
    if dtype == "float":
        return t.cast(pl.Float64, strict=False)
    if dtype == "date":
        return t.str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    if dtype == "bool_yn":
        return t.replace_strict(value_maps["yes_no"], default=None, return_dtype=pl.Boolean)
    if dtype == "survey_type_enum":
        st = value_maps["survey_type"]
        mapped = t.replace_strict(st["map"], default=st["other"], return_dtype=pl.String)
        return pl.when(t.is_null()).then(None).otherwise(mapped)
    if dtype == "enum":
        spec = value_maps["enums"][name]
        mapped = t.replace_strict(spec["map"], default=None, return_dtype=pl.String)
        return pl.when(t.is_null()).then(pl.lit(spec.get("blank"), pl.String)).otherwise(mapped)
    raise ValueError(f"no rule for dtype {dtype}")


def _invalid(name: str, dtype: str, value_maps: dict[str, Any]) -> pl.Expr:
    """Rows where a non-empty raw value did not survive its rule."""
    t, out = text(name), pl.col(name + "__out")
    if dtype == "survey_type_enum":
        return t.is_not_null() & ~t.is_in(list(value_maps["survey_type"]["map"]))
    if dtype == "enum":
        spec = value_maps["enums"][name]
        known = [*spec["map"], *spec.get("not_available", [])]
        return t.is_not_null() & ~t.is_in(known)
    return t.is_not_null() & out.is_null()


def normalize(
    frame: pl.DataFrame, dtypes: dict[str, str], value_maps: dict[str, Any]
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Apply every column's rule. Returns the typed frame and {column: invalid count} for counts > 0.
    Raises NormalizeError if any CCN is missing or malformed."""
    names = list(dtypes)
    typed = frame.with_columns([_rule(n, dtypes[n], value_maps).alias(n + "__out") for n in names])
    counts = typed.select([_invalid(n, dtypes[n], value_maps).sum().alias(n) for n in names]).row(
        0, named=True
    )
    for n in names:
        if dtypes[n] == "ccn":
            bad = typed.filter(~pl.col(n + "__out").fill_null("").str.contains(CCN_PATTERN))
            if bad.height:
                sample = bad[n].head(3).to_list()
                raise NormalizeError(
                    f"{bad.height} rows with a missing or malformed CCN, e.g. {sample}"
                )
    extra = typed.select(
        [
            text(n).is_in(value_maps["enums"][n].get("not_available", [])).alias(f"{n}__na")
            for n in names
            if dtypes[n] == "enum"
        ]
    )
    out = typed.select([pl.col(n + "__out").alias(n) for n in names]).hstack(extra)
    return out, {n: int(c) for n, c in counts.items() if c}


def null_column(name: str, dtype: str, height: int) -> pl.Series:
    """A column that does not exist yet in this vintage (e.g. turnover before 2022-01)."""
    return pl.Series(name, [None] * height, dtype=POLARS_TYPES[dtype])


def ownership_group(value_maps: dict[str, Any]) -> pl.Expr:
    """for_profit / non_profit / government from the part of ownership_type before " - "."""
    prefix = pl.col("ownership_type").str.split(" - ").list.first().str.strip_chars()
    return prefix.replace_strict(
        value_maps["ownership_group"], default=None, return_dtype=pl.String
    )


def measure_period_bounds(table: str) -> tuple[pl.Expr, pl.Expr]:
    """MDS `2024Q2-2025Q1` -> first day of 2024Q2, last day of 2025Q1; claims `20240101-20241231` -> dates."""
    mp = pl.col("measure_period").str.strip_chars()
    if table == "mds_qm":
        y1 = mp.str.extract(MDS_PERIOD, 1).cast(pl.Int32)
        q1 = mp.str.extract(MDS_PERIOD, 2).cast(pl.Int32)
        y2 = mp.str.extract(MDS_PERIOD, 3).cast(pl.Int32)
        q2 = mp.str.extract(MDS_PERIOD, 4).cast(pl.Int32)
        start = pl.date(y1, q1 * 3 - 2, 1)
        end = pl.date(y2, q2 * 3, 1).dt.month_end()
        return start, end
    start = mp.str.extract(CLAIMS_PERIOD, 1).str.strptime(pl.Date, "%Y%m%d", strict=False)
    end = mp.str.extract(CLAIMS_PERIOD, 2).str.strptime(pl.Date, "%Y%m%d", strict=False)
    return start, end
