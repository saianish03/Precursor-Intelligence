"""Lineage columns and reconciliation against the Step 1 inventory (docs/DATA_PIPELINE_SPEC.md §3.1 steps 6, 7)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

import polars as pl

from ..ingestion.layout import header_hash, normalize_header
from .reader import ParsedCsv


class ReconcileError(ValueError):
    """A file disagrees with its inventory row. A data failure (exit 2): nothing is written for it."""


def add_lineage(frame: pl.DataFrame, inv: Mapping[str, Any], ingested_at: datetime) -> pl.DataFrame:
    """Append the 8 lineage columns, all taken from the inventory row except the row number and load time."""
    processing_date = inv["processing_date"]
    return frame.with_columns(
        pl.lit(inv["vintage_month"], dtype=pl.String).alias("_vintage_month"),
        pl.lit(
            date.fromisoformat(processing_date) if processing_date else None, dtype=pl.Date
        ).alias("_processing_date"),
        pl.lit(inv["zip_name"], dtype=pl.String).alias("_source_zip"),
        pl.lit(inv["zip_sha256"], dtype=pl.String).alias("_zip_sha256"),
        pl.lit(inv["member_path"], dtype=pl.String).alias("_source_file"),
        pl.lit(inv["header_hash"], dtype=pl.String).alias("_header_hash"),
        pl.int_range(1, pl.len() + 1, dtype=pl.Int64).alias("_row_number"),
        pl.lit(ingested_at.astimezone(UTC), dtype=pl.Datetime("us", "UTC")).alias("_ingested_at"),
    )


def reconcile(parsed: ParsedCsv, inv: Mapping[str, Any]) -> None:
    """Raise unless the parsed file matches what Step 1 recorded: rows + blank rows == row_count, and the
    recomputed header hash equals the inventory's (same normalisation as Step 1, so a renamed or reordered
    column cannot slip through)."""
    total = parsed.frame.height + parsed.blank_rows
    if total != inv["row_count"]:
        raise ReconcileError(
            f"row count {parsed.frame.height} + {parsed.blank_rows} blank != inventory {inv['row_count']}"
        )
    actual = header_hash(normalize_header(parsed.frame.columns))
    if actual != inv["header_hash"]:
        raise ReconcileError(f"header hash {actual} != inventory {inv['header_hash']}")
