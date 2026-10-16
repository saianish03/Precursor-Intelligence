"""Content hash of a bronze file and the comparison with the previous vintage (docs/DATA_PIPELINE_SPEC.md §3.1 step 8)."""

from __future__ import annotations

import hashlib

import polars as pl

from ..common.io.metadata_store import MetadataStore

PUBLISHED = ("loaded", "reloaded")
_EXCLUDED = {"processing date"}
CHUNK_ROWS = 200_000


def content_hash(frame: pl.DataFrame, chunk_rows: int = CHUNK_ROWS) -> str:
    """sha256 over the data columns without `Processing Date`, rows sorted, so the same published data
    hashes the same in every vintage. Streamed in chunks: the largest file has millions of rows.
    Exact duplicate rows count (penalties keep them); a null and an empty string hash differently."""
    columns = [c for c in frame.columns if c.lower() not in _EXCLUDED]
    data = frame.select(columns).sort(columns, nulls_last=True)
    digest = hashlib.sha256()
    digest.update(data.head(0).write_csv().encode("utf-8"))
    for offset in range(0, data.height, chunk_rows):
        digest.update(
            data.slice(offset, chunk_rows).write_csv(include_header=False).encode("utf-8")
        )
    return digest.hexdigest()


def previous_content_hash(meta: MetadataStore, table: str, vintage: str) -> str | None:
    """The hash of the same table's nearest earlier published vintage, or None for its first one.
    Failed attempts never count, and a later attempt for the same vintage (a reload) wins."""
    rows = [
        r
        for r in meta.read("bronze_files", {"table_name": table, "status": list(PUBLISHED)})
        if r["vintage_month"] < vintage
    ]
    if not rows:
        return None
    latest = max(rows, key=lambda r: (r["vintage_month"], r["id"]))
    previous: str | None = latest["content_hash"]
    return previous


def content_unchanged(current: str, previous: str | None) -> bool | None:
    """Informational per-file flag; None when there is no earlier vintage to compare with."""
    return None if previous is None else current == previous
