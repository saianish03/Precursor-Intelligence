"""Representation and coverage (Lecture 4): how many rows and facilities each slice value has, and when."""

from __future__ import annotations

import polars as pl

from .config import BiasConfig
from .slices import SliceDef, group_columns, slice_col, slice_frame, with_slice_value


def representation(df: pl.DataFrame, slices: list[SliceDef], cfg: BiasConfig) -> pl.DataFrame:
    """One row per slice value: n_rows, n_facilities, shares, small_slice, first_t, last_t.

    Values listed in ``slices`` that have no rows are kept with zeros (and are small by definition).
    """
    total_rows, total_fac = df.height, df["ccn"].n_unique()
    parts = []
    for cols in group_columns(slices):
        g = df.group_by([slice_col(c) for c in cols]).agg(
            pl.len().alias("n_rows"),
            pl.col("ccn").n_unique().alias("n_facilities"),
            pl.col("t").min().alias("first_t"),
            pl.col("t").max().alias("last_t"),
        )
        parts.append(
            with_slice_value(g, cols).select(
                "slice", "value", "n_rows", "n_facilities", "first_t", "last_t"
            )
        )
    counts = pl.concat(parts) if parts else None
    out = slice_frame(slices)
    if counts is not None:
        out = out.join(counts, on=["slice", "value"], how="left")
    thr = cfg.thresholds
    return (
        out.with_columns(
            pl.col("n_rows").fill_null(0).cast(pl.Int64),
            pl.col("n_facilities").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            (pl.col("n_rows") / max(total_rows, 1)).alias("share_rows"),
            (pl.col("n_facilities") / max(total_fac, 1)).alias("share_facilities"),
            (
                (pl.col("n_facilities") < thr.min_slice_facilities)
                | (pl.col("n_rows") < thr.min_slice_rows)
            ).alias("small_slice"),
        )
        .sort("order")
        .select(
            "slice",
            "value",
            "n_rows",
            "n_facilities",
            "share_rows",
            "share_facilities",
            "small_slice",
            "first_t",
            "last_t",
        )
    )


def monthly_counts(df: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    """Rows per (single slice, value, t), for the slice-stability chart."""
    parts = [
        with_slice_value(df.group_by(slice_col(n), "t").agg(pl.len().alias("n_rows")), (n,)).select(
            "slice", "value", "t", "n_rows"
        )
        for n in names
    ]
    schema = {"slice": pl.String, "value": pl.String, "t": pl.String, "n_rows": pl.UInt32}
    if not parts:
        return pl.DataFrame(schema=schema)
    return pl.concat(parts).sort("slice", "value", "t")
