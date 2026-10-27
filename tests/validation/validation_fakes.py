"""Small clean tables for the validation tests (a module, not conftest, so imports never depend on order)."""

import io
from datetime import date

import polars as pl


def citations(n=4, vintage="2026-07", **cols):
    """n clean citation rows of one vintage; keyword arguments replace whole columns."""
    base = {
        "ccn": [f"0150{i:02d}" for i in range(n)],
        "survey_date": [date(2026, 5, 1 + i % 28) for i in range(n)],
        "scope_severity": ["D", "E", "G", "D"] * (n // 4) + ["D"] * (n % 4),
        "inspection_cycle": [1, 2, 3, 1] * (n // 4) + [1] * (n % 4),
        "is_complaint": [True, False] * (n // 2) + [False] * (n % 2),
        "vintage_month": [vintage] * n,
    }
    return pl.DataFrame(base | cols)


def parquet(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()
