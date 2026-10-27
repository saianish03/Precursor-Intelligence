"""Infer a table's schema from its months and keep it as a hand-editable baseline, ``configs/contracts/<table>.yaml``.

Inferred per column: dtype, nullable (any null in any month), min/max for numbers, and the allowed values of a text
column with few distinct values. Edit the file by hand to tighten a rule (e.g. ``allowed`` severities, or
``str_length: 6`` for a CCN); `validate --write-schema` overwrites it, so commit hand edits first.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..common.config import ConfigError

Scalar = bool | int | float | str


class ColumnRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dtype: str
    nullable: bool = True
    min: int | float | None = None
    max: int | float | None = None
    allowed: list[Scalar] | None = None
    allowed_severity: Literal["warning"] | None = None  # outside `allowed`: a warning, not an error
    str_length: int | None = Field(default=None, gt=0)  # exact length of every non-null value


class TableSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str
    layer: str
    inferred_from: dict[str, Any] = {}  # first and last month, number of months
    columns: dict[str, ColumnRule]


class _Column:
    """What has been seen of one column so far."""

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype
        self.nullable = False
        self.min: Any = None
        self.max: Any = None
        self.values: set[Any] | None = set()  # None once there are too many to list


def infer_schema(
    months: Iterable[tuple[str, pl.DataFrame]], *, table: str, layer: str, low_cardinality_max: int
) -> TableSchema:
    """One pass over ``(month, frame)`` pairs, so a large table never has to be in memory at once."""
    cols: dict[str, _Column] = {}
    seen: list[str] = []
    for month, df in months:
        seen.append(month)
        for name in df.columns:
            s = df[name]
            col = cols.setdefault(name, _Column(str(s.dtype)))
            col.nullable |= s.null_count() > 0
            values = s.drop_nulls()
            if s.dtype.is_numeric() and len(values):
                lo, hi = values.min(), values.max()
                col.min = lo if col.min is None else min(col.min, lo)
                col.max = hi if col.max is None else max(col.max, hi)
            if s.dtype == pl.String and col.values is not None:
                col.values |= set(values.unique().to_list())
                if len(col.values) > low_cardinality_max:
                    col.values = None
    rules: dict[str, ColumnRule] = {}
    for name, col in cols.items():
        rule = ColumnRule(dtype=col.dtype, nullable=col.nullable, min=col.min, max=col.max)
        if col.dtype == "String" and col.values:
            rule.allowed = sorted(col.values)
        rules[name] = rule
    first, last = (min(seen), max(seen)) if seen else (None, None)
    return TableSchema(
        table=table,
        layer=layer,
        inferred_from={"first": first, "last": last, "months": len(seen)},
        columns=rules,
    )


def dump_schema(schema: TableSchema) -> str:
    """YAML with one flow-style line per column, so a rule is easy to find and edit."""
    lines = [
        f"# Baseline schema for {schema.layer}/{schema.table} (precursor-pipeline validate).",
        "# Inferred by `validate --write-schema`; tighten by hand (allowed, min, max, str_length, nullable).",
        f"table: {schema.table}",
        f"layer: {schema.layer}",
        "inferred_from: "
        + yaml.safe_dump(schema.inferred_from, default_flow_style=True, sort_keys=False).strip(),
        "columns:",
    ]
    for name, rule in schema.columns.items():
        body = rule.model_dump(exclude_none=True)
        flow = yaml.safe_dump(body, default_flow_style=True, sort_keys=False, width=10**6)
        lines.append(f"  {name}: {flow.strip()}")
    return "\n".join(lines) + "\n"


def write_schema(schema: TableSchema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_schema(schema), encoding="utf-8")


def load_schema(path: Path) -> TableSchema:
    if not path.is_file():
        raise ConfigError(f"schema file not found: {path} (make it with `validate --write-schema`)")
    try:
        return TableSchema.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except Exception as exc:  # pydantic.ValidationError and yaml errors
        raise ConfigError(f"{path}: {exc}") from exc
