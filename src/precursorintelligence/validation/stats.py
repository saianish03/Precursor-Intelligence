"""Statistics of one month of one table, as plain JSON-ready dicts.

Per table: row count. Per column: dtype, null %, distinct count; min/max/mean/std for numbers; min/max for dates;
the most frequent values for text and booleans.
"""

from __future__ import annotations

from typing import Any

import polars as pl


def _json_value(v: Any) -> Any:
    return v if v is None or isinstance(v, (bool, int, float, str)) else str(v)


def column_stats(s: pl.Series, top_values: int) -> dict[str, Any]:
    n = s.len()
    out: dict[str, Any] = {
        "dtype": str(s.dtype),
        "null_pct": round(100 * s.null_count() / n, 4) if n else 0.0,
        "distinct": s.drop_nulls().n_unique(),
    }
    values = s.drop_nulls()
    if s.dtype.is_numeric():
        out |= {
            "min": _json_value(values.min()) if len(values) else None,
            "max": _json_value(values.max()) if len(values) else None,
            "mean": values.mean() if len(values) else None,
            "std": values.std() if len(values) > 1 else None,
        }
    elif s.dtype.is_temporal():
        out |= {
            "min": _json_value(values.min()) if len(values) else None,
            "max": _json_value(values.max()) if len(values) else None,
        }
    else:
        counts = values.value_counts(sort=True).head(top_values)
        out["top_values"] = [[_json_value(v), int(c)] for v, c in counts.iter_rows()]
    return out


def table_stats(df: pl.DataFrame, *, table: str, month: str, top_values: int) -> dict[str, Any]:
    return {
        "table": table,
        "month": month,
        "row_count": df.height,
        "columns": {c: column_stats(df[c], top_values) for c in df.columns},
    }
