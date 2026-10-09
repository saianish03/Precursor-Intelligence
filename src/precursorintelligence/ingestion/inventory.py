"""Task ``inventory`` (fan-out: one call per stored zip).

Writes one ``inventory`` row per non-junk zip member. For extracted (in-scope) CSVs it also records
the Processing Date, the vintage month, the full row count, the column count and a header hash.

Vintage rule (guide §2.5): vintage_month = YYYY-MM(Processing Date). Fallbacks, used only when the
column is missing (L1 files), are logged and recorded in ``vintage_source``:
    processing_date  ->  filename_token (MonYYYY in the file name)  ->  zip_date
"""

from __future__ import annotations

import csv
import io
import json
import logging
import zipfile
from collections.abc import Iterator

import polars as pl

from ..common.context import Context, utc_now
from .download import last_good_download
from .layout import (
    OTHER,
    TableClassifier,
    detect_layout,
    filename_month,
    header_hash,
    manifest_filenames,
    processing_date_from,
    read_header_and_first_row,
)
from .sources.base import CatalogEntry
from .zipsafe import inspect_archive

log = logging.getLogger(__name__)

JUNK = "junk"


def count_rows(ctx: Context, key: str) -> int:
    """Data rows (header excluded). Uses Polars on local files, a streaming csv reader elsewhere."""
    path = ctx.object_store.local_path(key)
    if path is not None:
        return int(pl.scan_csv(path, infer_schema=False, encoding="utf8-lossy", truncate_ragged_lines=True,
                               quote_char='"').select(pl.len()).collect().item())
    with ctx.object_store.open_read(key) as f:
        text = io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace", newline="")
        return max(0, sum(1 for _ in csv.reader(text)) - 1)


def _manifest_names(ctx: Context, zip_key: str, members) -> set[str] | None:
    man = [m for m in members if not m.is_junk and m.file_name == "manifest.json"]
    if not man:
        return None
    with ctx.object_store.open_read(zip_key) as f, zipfile.ZipFile(f) as zf:
        return manifest_filenames(zf.read(man[0].path))


def inventory_entry(ctx: Context, entry: CatalogEntry, zip_key: str, zip_sha256: str) -> list[dict]:
    classifier = TableClassifier(ctx.cfg.tables)
    with ctx.object_store.open_read(zip_key) as f:
        members = inspect_archive(f, ctx.cfg.security.zip)
    real = [m for m in members if not m.is_junk]
    layout = detect_layout([m.file_name for m in real])
    manifest = _manifest_names(ctx, zip_key, members)
    extracted = {r["member_path"]: r for r in ctx.metadata.read("extracted_files", {"entry_id": entry.entry_id})
                 if r["zip_sha256"] == zip_sha256}

    rows = []
    for m in members:
        if m.is_junk:  # recorded (they explain bloated zips) but never extracted or used
            rows.append({
                "entry_id": entry.entry_id, "member_path": m.path, "zip_name": zip_key.rsplit("/", 1)[-1],
                "zip_date": entry.zip_date.isoformat(), "zip_type": entry.type, "zip_sha256": zip_sha256,
                "layout": layout, "file_name": m.file_name, "table_name": JUNK, "file_size_bytes": m.size,
                "compressed_size_bytes": m.compressed_size, "crc32": m.crc32, "status": JUNK,
                "run_id": ctx.run_id, "inventoried_at": utc_now(),
            })
            continue
        table = classifier.classify(m.file_name)
        row = {
            "entry_id": entry.entry_id, "member_path": m.path, "zip_name": zip_key.rsplit("/", 1)[-1],
            "zip_date": entry.zip_date.isoformat(), "zip_type": entry.type, "zip_sha256": zip_sha256,
            "layout": layout, "file_name": m.file_name, "table_name": table,
            "file_size_bytes": m.size, "compressed_size_bytes": m.compressed_size, "crc32": m.crc32,
            "in_manifest": (m.file_name in manifest or m.file_name.split("_", 2)[-1] in manifest)
            if manifest is not None and m.extension == ".csv" else None,
            "status": OTHER if table == OTHER else "pending", "run_id": ctx.run_id, "inventoried_at": utc_now(),
        }
        ext = extracted.get(m.path)
        if ext is not None and m.extension == ".csv":
            row["extracted_key"] = ext["object_key"]
            with ctx.object_store.open_read(ext["object_key"]) as fh:
                header, first = read_header_and_first_row(fh)
            row.update(n_columns=len(header), header_hash=header_hash(header),
                       header_json=json.dumps(header, ensure_ascii=False))
            pdate = processing_date_from(header, first)
            row["row_count"] = count_rows(ctx, ext["object_key"])
        else:
            pdate = None
        if pdate:
            row.update(processing_date=pdate.isoformat(), vintage_month=pdate.strftime("%Y-%m"),
                       vintage_source="processing_date")
        elif (tok := filename_month(m.file_name)) is not None:
            row.update(vintage_month=tok, vintage_source="filename_token")
            if table != OTHER:
                expected = not ctx.cfg.tables[table].has_processing_date
                log.log(logging.DEBUG if expected else logging.WARNING,
                        "%s: no Processing Date; vintage %s taken from file name", m.file_name, tok)
        elif m.extension == ".csv":
            row.update(vintage_month=entry.zip_date.strftime("%Y-%m"), vintage_source="zip_date")
            if table != OTHER:
                log.warning("%s: no Processing Date or name token; vintage falls back to zip date", m.file_name)
        rows.append(row)

    # Inventory is a full snapshot of this zip: replace any rows from earlier runs.
    ctx.metadata.delete_where("inventory", {"entry_id": entry.entry_id})
    ctx.metadata.upsert("inventory", rows)
    n_scope = sum(r["table_name"] not in (OTHER, JUNK) for r in rows)
    log.info("inventory %s: %d members (%d in scope), layout %s, vintages %s", entry.zip_name, len(real), n_scope,
             layout, sorted({r.get("vintage_month") for r in rows if r["table_name"] not in (OTHER, JUNK) and r.get("vintage_month")}))
    return rows


def iter_good_downloads(ctx: Context, entries: list[CatalogEntry]) -> Iterator[tuple[CatalogEntry, dict]]:
    """Latest successful download_log row per entry (the object downstream tasks should read)."""
    for e in entries:
        row = last_good_download(ctx, e.entry_id)
        if row is None:
            log.warning("no successful download for %s; skipping downstream tasks", e.entry_id)
            continue
        yield e, row
