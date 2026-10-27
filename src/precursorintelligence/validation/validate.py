"""Check one month of a table: against its schema YAML with Pandera (errors), and against the previous month's
statistics (warnings)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandera.polars as pa
import polars as pl

from ..common.config import ConfigError
from .schema import ColumnRule, TableSchema

ERROR, WARNING = "error", "warning"
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
    if rule.allowed is not None:
        checks.append(pa.Check.isin(rule.allowed))
    if rule.str_length is not None:
        checks.append(pa.Check.str_length(rule.str_length, rule.str_length))
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
    }.get(check)


def check_schema(df: pl.DataFrame, schema: TableSchema, *, month: str) -> list[Anomaly]:
    """Every schema violation in the month, one anomaly per (column, check). All are errors."""
    try:
        pandera_schema(schema).validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases.with_columns(pl.col("failure_case").cast(pl.String))
    else:
        return []
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
    for column, pandera_check, n, examples in groups.iter_rows():
        check = _check_name(pandera_check)
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
    return out
