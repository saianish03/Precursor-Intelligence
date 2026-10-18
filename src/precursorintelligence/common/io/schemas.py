"""Bookkeeping tables written by Step 1 (SQLAlchemy Core; portable across SQLite and Postgres).

Dates/timestamps are stored as ISO-8601 strings so the same rows round-trip through any backend
(SQLite, Postgres, BigQuery) without timezone or type surprises.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Boolean, Column, Integer, MetaData, String, Table, Text

metadata = MetaData()

# One row per archive entry advertised by the CMS catalog endpoints.
catalog = Table(
    "catalog",
    metadata,
    Column("entry_id", String(255), primary_key=True),  # f"{type}:{zip_date}:{zip_name}"
    Column("name", Text),
    Column("nid", String(32)),
    Column("type", String(32)),
    Column("theme", String(64)),
    Column("zip_date", String(10)),
    Column("zip_name", String(255)),
    Column("url", Text),
    Column("size_bytes", BigInteger),
    Column("source_endpoint", String(32)),
    Column("first_seen_at", String(32)),
    Column("last_seen_at", String(32)),
    Column("active", Boolean),
)

# Release metadata of the in-scope datasets (documented metastore API; current release only).
source_datasets = Table(
    "source_datasets",
    metadata,
    Column("dataset_id", String(32), primary_key=True),
    Column("table_name", String(64)),
    Column("title", Text),
    Column("modified", String(32)),
    Column("released", String(32)),
    Column("next_update", String(32)),
    Column("current_download_url", Text),
    Column("fetched_at", String(32)),
)

# Append-only audit log of every download decision.
download_log = Table(
    "download_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64)),
    Column("entry_id", String(255)),
    Column("zip_name", String(255)),
    Column("zip_date", String(10)),
    Column("type", String(32)),
    Column("url", Text),
    Column("catalog_size", BigInteger),
    Column("expected_size", BigInteger),
    Column("actual_size", BigInteger),
    Column("sha256", String(64)),
    Column("http_last_modified", String(64)),
    Column("http_content_type", String(128)),
    Column("object_key", Text),
    Column("object_uri", Text),
    Column(
        "status", String(32)
    ),  # downloaded|skipped|unchanged|republished|adopted|quarantined|failed|rejected
    Column("detail", Text),
    Column("logged_at", String(32)),
)

# Members extracted from each zip.
extracted_files = Table(
    "extracted_files",
    metadata,
    Column("entry_id", String(255), primary_key=True),
    Column("member_path", String(512), primary_key=True),
    Column("zip_sha256", String(64)),
    Column("table_name", String(64)),
    Column("object_key", Text),
    Column("size_bytes", BigInteger),
    Column("crc32", String(8)),
    Column("run_id", String(64)),
    Column("extracted_at", String(32)),
)

# One row per (zip, member): the Step 1 inventory (guide §2.6).
inventory = Table(
    "inventory",
    metadata,
    Column("entry_id", String(255), primary_key=True),
    Column("member_path", String(512), primary_key=True),
    Column("zip_name", String(255)),
    Column("zip_date", String(10)),
    Column("zip_type", String(32)),
    Column("zip_sha256", String(64)),
    Column("layout", String(4)),
    Column("file_name", String(255)),
    Column("table_name", String(64)),
    Column("extracted_key", Text),
    Column("processing_date", String(10)),
    Column("vintage_month", String(7)),
    Column("vintage_source", String(32)),
    Column("row_count", BigInteger),
    Column("n_columns", Integer),
    Column("header_hash", String(16)),
    Column("header_json", Text),
    Column("file_size_bytes", BigInteger),
    Column("compressed_size_bytes", BigInteger),
    Column("crc32", String(8)),
    Column("in_manifest", Boolean),
    # pending (inventoried, not yet resolved) | selected | duplicate_vintage | out_of_scope | other |
    # junk | unresolved. Downstream reads only `selected`, and only when no row is `pending`.
    Column("status", String(32)),
    Column("status_detail", Text),
    Column("run_id", String(64)),
    Column("inventoried_at", String(32)),
)

anomalies = Table(
    "anomalies",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64)),
    Column("entry_id", String(255)),
    Column("kind", String(64)),
    Column("severity", String(16)),
    Column("detail", Text),
    Column("created_at", String(32)),
)

# Task-level audit trail (start/finish, counts, failures); also the idempotency marker store.
run_log = Table(
    "run_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64)),
    Column("task", String(64)),
    Column("entry_id", String(255)),
    Column("status", String(32)),
    Column("started_at", String(32)),
    Column("finished_at", String(32)),
    Column("duration_s", String(16)),
    Column("detail", Text),
)

# Run lock (decision D8): at most one row per lock name; the primary key makes acquiring atomic.
run_lock = Table(
    "run_lock",
    metadata,
    Column("lock_name", String(64), primary_key=True),
    Column("run_id", String(64)),
    Column("holder", String(255)),  # host:pid of the process that acquired it
    Column("acquired_at", String(32)),
)

# Bronze (spec §3.1 step 9): one row per file bronze loaded or failed; skipped files get no row.
# Append-only, so a failed attempt stays visible after a later successful load.
bronze_files = Table(
    "bronze_files",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64)),
    Column("table_name", String(64)),
    Column("vintage_month", String(7)),
    Column("entry_id", String(255)),
    Column("member_path", String(512)),
    Column("zip_sha256", String(64)),
    Column("header_hash", String(16)),
    Column("status", String(16)),  # loaded | reloaded | failed
    Column("row_count", BigInteger),
    Column("blank_rows", BigInteger),
    Column("encoding", String(16)),  # utf-8 | cp1252
    Column("content_hash", String(64)),
    Column("content_unchanged", Boolean),  # NULL for a table's first vintage
    Column("object_uri", Text),
    Column("detail", Text),
    Column("loaded_at", String(32)),
)

# Bronze release flags (spec §3.1 steps 8 and "Republished data"): the current value per flag.
# `table_name` is '' for the release-level stale_refresh flag, so it can be part of the key.
vintage_flags = Table(
    "vintage_flags",
    metadata,
    Column("vintage_month", String(7), primary_key=True),
    Column("flag", String(32), primary_key=True),  # stale_refresh | revised_after_publication
    Column("table_name", String(64), primary_key=True),
    Column("value", Boolean),
    Column("detail", Text),
    Column("run_id", String(64)),
    Column("updated_at", String(32)),
)

TABLES = {t.name: t for t in metadata.sorted_tables}
