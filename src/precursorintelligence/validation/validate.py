"""Check one month of a table: against its schema YAML with Pandera (errors), and against the previous month's
statistics (warnings)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from fnmatch import fnmatch
from typing import Any

import pandera.polars as pa
import polars as pl

from ..common.config import ConfigError, DriftConfig
from .schema import ColumnRule, TableSchema

ERROR, WARNING = "error", "warning"
MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")  # YYYY-MM, as the contracts' month patterns
_EXAMPLES = 5  # failing values kept per check


@dataclass(frozen=True)
class Anomaly:
    table: str
    month: str
    check: str  # missing_column, unexpected_column, wrong_dtype, null_values, out_of_range, ...
    severity: str  # error | warning
    column: str | None
    observed: Any
    expected: Any
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dtype(name: str) -> type[pl.DataType]:
    dt = getattr(pl, name, None)
    if not (isinstance(dt, type) and issubclass(dt, pl.DataType)):
        raise ConfigError(f"unknown dtype in schema: {name!r}")
    return dt


def _checks(rule: ColumnRule) -> list[pa.Check]:
    checks: list[pa.Check] = []
    if rule.min is not None and rule.max is not None:
        checks.append(pa.Check.in_range(rule.min, rule.max))
    elif rule.min is not None:
        checks.append(pa.Check.ge(rule.min))
    elif rule.max is not None:
        checks.append(pa.Check.le(rule.max))
    if rule.allowed is not None and rule.allowed_severity is None:
        checks.append(pa.Check.isin(rule.allowed))
    if rule.str_length is not None:
        checks.append(pa.Check.str_length(rule.str_length, rule.str_length))
    if rule.pattern is not None:
        checks.append(pa.Check.str_matches(rule.pattern))
    return checks


def pandera_schema(schema: TableSchema) -> pa.DataFrameSchema:
    columns = {
        name: pa.Column(_dtype(rule.dtype), checks=_checks(rule), nullable=rule.nullable)
        for name, rule in schema.columns.items()
    }
    return pa.DataFrameSchema(columns, strict=True, name=schema.table)


def _check_name(pandera_check: str) -> str:
    for prefix, name in (
        ("column_in_dataframe", "missing_column"),
        ("column_in_schema", "unexpected_column"),
        ("dtype(", "wrong_dtype"),
        ("not_nullable", "null_values"),
        ("in_range", "out_of_range"),
        ("greater_than", "out_of_range"),
        ("less_than", "out_of_range"),
        ("isin", "allowed_values"),
        ("str_length", "str_length"),
        ("str_matches", "pattern"),
    ):
        if pandera_check.startswith(prefix):
            return name
    return pandera_check


def _expected(check: str, rule: ColumnRule | None) -> Any:
    if rule is None:
        return None
    return {
        "wrong_dtype": rule.dtype,
        "null_values": "no nulls",
        "out_of_range": [rule.min, rule.max],
        "allowed_values": rule.allowed,
        "str_length": rule.str_length,
        "pattern": rule.pattern,
    }.get(check)


def _new_values(df: pl.DataFrame, schema: TableSchema, month: str) -> list[Anomaly]:
    """Values outside a warning-only `allowed` list (e.g. a new CMS ownership type): one warning per column."""
    out = []
    for name, rule in schema.columns.items():
        if rule.allowed_severity != WARNING or rule.allowed is None or name not in df.columns:
            continue
        if str(df[name].dtype) != rule.dtype:
            continue  # reported as wrong_dtype
        bad = df[name].drop_nulls().filter(~df[name].drop_nulls().is_in(rule.allowed))
        if bad.len():
            shown = [str(v) for v in bad.unique(maintain_order=True).head(_EXAMPLES)]
            out.append(
                Anomaly(schema.table, month, "allowed_values", WARNING, name,
                        {"rows": bad.len(), "examples": shown}, rule.allowed,
                        f"{bad.len()} rows have a value not listed, e.g. {', '.join(shown)}")
            )  # fmt: skip
    return out


def _duplicate_keys(df: pl.DataFrame, schema: TableSchema, month: str) -> list[Anomaly]:
    key = schema.unique
    if not key or not set(key) <= set(df.columns):
        return []  # a missing key column is reported as missing_column
    dup = df.select(key).filter(df.select(key).is_duplicated())
    if not dup.height:
        return []
    shown = [
        "|".join(str(v) for v in row)
        for row in dup.unique(maintain_order=True).head(_EXAMPLES).rows()
    ]
    return [
        Anomaly(schema.table, month, "duplicate_key", ERROR, ",".join(key),
                {"rows": dup.height, "examples": shown}, key,
                f"{dup.height} rows share a key ({', '.join(key)}), e.g. {', '.join(shown)}")
    ]  # fmt: skip


UNLABELLED = ["censored", "immature"]  # gold/labels.py: y is null for exactly these


def _rule_violations(rule: str, n: str) -> tuple[pl.Expr, str]:
    """(rows breaking ``rule`` for label ``n``, what should hold): gold/labels.py and gold/splits.py."""
    y, status, split = pl.col(f"y_{n}"), pl.col(f"status_{n}"), pl.col(f"split_{n}")
    if rule == "y_null_iff_unlabelled":
        return y.is_null() != status.is_in(UNLABELLED), "y null exactly when censored or immature"
    if rule == "y_matches_status":
        bad = (status.eq_missing("positive") != y.eq_missing(1)) | (
            status.eq_missing("negative") != y.eq_missing(0)
        )
        return bad, "y = 1 exactly when positive, y = 0 exactly when negative"
    return split.eq_missing("score") != status.eq_missing("immature"), "score exactly when immature"


def _rules(df: pl.DataFrame, schema: TableSchema, month: str) -> list[Anomaly]:
    out: list[Anomaly] = []
    labels = [c.removeprefix("status_") for c in df.columns if c.startswith("status_")]
    for rule in schema.rules or []:
        for n in labels:
            needed = [f"y_{n}", f"status_{n}"]
            needed += [f"split_{n}"] if rule == "score_iff_immature" else []
            if not set(needed) <= set(df.columns):
                continue  # reported as missing_column
            bad_rows, should = _rule_violations(rule, n)
            bad = df.filter(bad_rows).select(needed)
            if not bad.height:
                continue
            shown = [
                " ".join(f"{c}={v}" for c, v in zip(needed, row, strict=True))
                for row in bad.unique(maintain_order=True).head(_EXAMPLES).rows()
            ]
            out.append(
                Anomaly(schema.table, month, rule, ERROR, f"status_{n}",
                        {"rows": bad.height, "examples": shown}, should,
                        f"{bad.height} rows break {rule} ({should}), e.g. {shown[0]}")
            )  # fmt: skip
    return out


def check_schema(df: pl.DataFrame, schema: TableSchema, *, month: str) -> list[Anomaly]:
    """Every schema violation in the month, one anomaly per (column, check), and an error for a month with no
    rows. All are errors, except a value outside an `allowed` list marked ``allowed_severity: warning``."""
    empty = [
        Anomaly(schema.table, month, "empty_month", ERROR, None, 0, "at least one row",
                f"{schema.table} {month} has no rows")
    ] if not df.height else []  # fmt: skip
    extra = (
        empty
        + _duplicate_keys(df, schema, month)
        + _rules(df, schema, month)
        + _new_values(df, schema, month)
    )
    try:
        pandera_schema(schema).validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases.with_columns(pl.col("failure_case").cast(pl.String))
    else:
        return extra
    out: list[Anomaly] = []
    observed: Any
    table_level = pl.col("schema_context") == "DataFrameSchema"
    # Table-level cases (a missing or unexpected column) name the column in failure_case: one anomaly each.
    for name, pandera_check in (
        cases.filter(table_level).select("failure_case", "check").iter_rows()
    ):
        check = _check_name(pandera_check)
        missing = check == "missing_column"
        detail = f"column {name} is missing" if missing else f"column {name} is not in the schema"
        expected = name if missing else None
        observed = None if missing else name
        out.append(Anomaly(schema.table, month, check, ERROR, name, observed, expected, detail))
    groups = (
        cases.filter(~table_level)
        .group_by("column", "check", maintain_order=True)
        .agg(
            pl.len().alias("n"), pl.col("failure_case").unique(maintain_order=True).head(_EXAMPLES)
        )
    )
    mistyped = {
        c for c, k in groups.select("column", "check").iter_rows() if k.startswith("dtype(")
    }
    for column, pandera_check, n, examples in groups.iter_rows():
        check = _check_name(pandera_check)
        if column in mistyped and check != "wrong_dtype":
            continue  # its other checks failed only because of the dtype
        rule = schema.columns.get(column)
        if check == "wrong_dtype":
            observed = examples[0]
            detail = f"{column} is {examples[0]}, expected {rule and rule.dtype}"
        else:
            shown = [e for e in examples if e is not None]
            observed = {"rows": n, "examples": shown}
            detail = f"{n} rows fail {pandera_check}" + (
                f", e.g. {', '.join(shown)}" if shown else ""
            )
        out.append(
            Anomaly(
                schema.table, month, check, ERROR, column, observed, _expected(check, rule), detail
            )
        )
    return out + extra


def check_month(
    df: pl.DataFrame, *, table: str, month: str, first_month: str | None = None
) -> list[Anomaly]:
    """The rows belong to the month they are stored under: ``vintage_month`` equals it, ``cutoff_date`` is its
    first day, and the month is not before ``first_month`` (gold: spine_start). All are errors. A month that is not
    YYYY-MM (a gold ``t`` of 2026-13, a silver folder ``vintage=junk``) is one ``bad_month`` error, and nothing else
    is compared with it."""
    if not MONTH.fullmatch(month):
        return [
            Anomaly(table, month, "bad_month", ERROR, None, month, "YYYY-MM",
                    f"month {month!r} is not YYYY-MM")
        ]  # fmt: skip
    out = []
    if first_month is not None and month < first_month:
        out.append(
            Anomaly(table, month, "before_first_month", ERROR, None, month, first_month,
                    f"month {month} is before {first_month}")
        )  # fmt: skip
    first_day = date(int(month[:4]), int(month[5:7]), 1)
    for name, expected, dtype in (
        ("vintage_month", month, pl.String),
        ("cutoff_date", first_day, pl.Date),
    ):
        if name not in df.columns or df[name].dtype != dtype:
            continue  # missing or mistyped: reported by check_schema
        bad = df[name].drop_nulls().filter(df[name].drop_nulls() != expected)
        if bad.len():
            shown = [str(v) for v in bad.unique(maintain_order=True).head(_EXAMPLES)]
            out.append(
                Anomaly(table, month, "month_mismatch", ERROR, name,
                        {"rows": bad.len(), "examples": shown}, str(expected),
                        f"{bad.len()} rows of {month} have {name} {', '.join(shown)}, expected {expected}")
            )  # fmt: skip
    return out


def check_drift(
    stats: dict[str, Any], prev: dict[str, Any], cfg: DriftConfig, *, table: str
) -> list[Anomaly]:
    """This month's stats against the previous month's (both from stats.table_stats). All are warnings."""
    month, out = stats["month"], []
    rows, prev_rows = stats["row_count"], prev["row_count"]
    if prev_rows:
        change = 100 * (rows - prev_rows) / prev_rows
        if abs(change) > cfg.row_count_change_pct:
            out.append(
                Anomaly(table, month, "row_count_change", WARNING, None, rows, prev_rows,
                        f"{rows} rows vs {prev_rows} in {prev['month']} ({change:+.1f}%, "
                        f"limit {cfg.row_count_change_pct}%)")
            )  # fmt: skip
    for name, col in stats["columns"].items():
        before = prev["columns"].get(name)
        if before is None or any(fnmatch(name, p) for p in cfg.skip_columns):
            continue
        jump = col["null_pct"] - before["null_pct"]
        if jump > cfg.null_rate_jump_pts:
            out.append(
                Anomaly(table, month, "null_rate_jump", WARNING, name, col["null_pct"], before["null_pct"],
                        f"{name} null {col['null_pct']:.2f}% vs {before['null_pct']:.2f}% "
                        f"(+{jump:.2f} points, limit {cfg.null_rate_jump_pts})")
            )  # fmt: skip
        mean, prev_mean, prev_std = col.get("mean"), before.get("mean"), before.get("std")
        if mean is None or prev_mean is None or not prev_std:
            continue
        shift = abs(mean - prev_mean) / prev_std
        if shift > cfg.mean_shift_std:
            out.append(
                Anomaly(table, month, "mean_shift", WARNING, name, mean, prev_mean,
                        f"{name} mean {mean:.4g} vs {prev_mean:.4g} ({shift:.2f} previous std, "
                        f"limit {cfg.mean_shift_std})")
            )  # fmt: skip
    return out
