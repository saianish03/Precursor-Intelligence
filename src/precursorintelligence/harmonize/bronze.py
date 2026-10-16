"""Load one selected CMS file into a lossless all-text bronze partition (docs/DATA_PIPELINE_SPEC.md §3.1)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from ..common.io.metadata_store import MetadataStore
from ..common.io.object_store import ObjectStore, ObjectStoreError
from ..common.io.table_sink import TableSink
from .content import PUBLISHED, content_hash, content_unchanged, previous_content_hash
from .flags import mark_revised
from .lineage import ReconcileError, add_lineage, reconcile
from .reader import DecodeError, decode_csv, parse_csv_text

log = logging.getLogger(__name__)

LOADED, RELOADED, FAILED, SKIPPED = "loaded", "reloaded", "failed", "skipped"
# What marks one file as bad data rather than a bug: nothing is written for it and the run exits 2.
FILE_ERRORS = (
    DecodeError,
    ReconcileError,
    pl.exceptions.PolarsError,
    ObjectStoreError,
    FileNotFoundError,
)


@dataclass(frozen=True)
class FileResult:
    table: str
    vintage: str
    status: str  # loaded | reloaded | failed | skipped
    detail: str = ""
    revised: bool = False  # a reload whose content differs from what was published before


def _partition(inv: Mapping[str, Any]) -> dict[str, str]:
    return {"vintage": inv["vintage_month"]}


def _published_row(meta: MetadataStore, inv: Mapping[str, Any]) -> dict[str, Any] | None:
    rows = meta.read(
        "bronze_files",
        {
            "table_name": inv["table_name"],
            "vintage_month": inv["vintage_month"],
            "status": list(PUBLISHED),
        },
    )
    return max(rows, key=lambda r: r["id"]) if rows else None


def _partition_exists(sink: TableSink, inv: Mapping[str, Any]) -> bool:
    return _partition(inv) in sink.list_partitions(inv["table_name"])


def load_file(
    store: ObjectStore,
    meta: MetadataStore,
    sink: TableSink,
    inv: Mapping[str, Any],
    *,
    run_id: str,
    now: datetime,
) -> FileResult:
    """Skip, load, reload or fail one file. A file that fails any check writes no partition (the old one,
    if any, stays) and gets a `failed` row; it is retried next run because only `loaded` and `reloaded`
    rows count as done. A bug (anything outside FILE_ERRORS) propagates."""
    table, vintage = inv["table_name"], inv["vintage_month"]
    previous = _published_row(meta, inv)
    exists = _partition_exists(sink, inv)
    if (
        previous is not None
        and exists
        and previous["zip_sha256"] == inv["zip_sha256"]
        and previous["header_hash"] == inv["header_hash"]
    ):
        return FileResult(table, vintage, SKIPPED)

    row: dict[str, Any] = {
        "run_id": run_id,
        "table_name": table,
        "vintage_month": vintage,
        "entry_id": inv["entry_id"],
        "member_path": inv["member_path"],
        "zip_sha256": inv["zip_sha256"],
        "header_hash": inv["header_hash"],
        "loaded_at": now.isoformat(timespec="seconds"),
    }
    try:
        with store.open_read(inv["extracted_key"]) as f:
            raw = f.read()
        text, encoding = decode_csv(raw)
        parsed = parse_csv_text(text)
        reconcile(parsed, inv)
        digest = content_hash(parsed.frame)
        frame = add_lineage(parsed.frame, inv, now)
    except FILE_ERRORS as exc:
        log.warning("bronze failed for %s %s: %s", table, vintage, exc)
        meta.append("bronze_files", [row | {"status": FAILED, "detail": str(exc)}])
        return FileResult(table, vintage, FAILED, str(exc))

    uri = sink.write_partition(table, _partition(inv), frame, overwrite=True)
    status = RELOADED if (previous is not None or exists) else LOADED
    meta.append(
        "bronze_files",
        [
            row
            | {
                "status": status,
                "row_count": parsed.frame.height,
                "blank_rows": parsed.blank_rows,
                "encoding": encoding,
                "content_hash": digest,
                "content_unchanged": content_unchanged(
                    digest, previous_content_hash(meta, table, vintage)
                ),
                "object_uri": uri,
            }
        ],
    )
    revised = status == RELOADED and previous is not None and previous["content_hash"] != digest
    if revised:
        mark_revised(meta, table, vintage, run_id, now)
    log.info("bronze %s %s %s (%d rows)", status, table, vintage, parsed.frame.height)
    return FileResult(table, vintage, status, revised=revised)
