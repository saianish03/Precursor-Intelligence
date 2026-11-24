"""The read-only copy of the metadata store in BigQuery (docs/DATA_PIPELINE_SPEC.md §0.6, §15.3).

``precursor-pipeline mirror-metadata`` runs at the end of each DAG run and replaces every table of the env's
``bigquery.meta_dataset`` with the store's current rows: one load job per table (never streaming, D6),
``WRITE_TRUNCATE`` on the whole table, created if missing. Each table's schema comes from its columns here, so an
empty table still has its columns, and a column Postgres gains reaches BigQuery at the next mirror. ``run_lock`` stays
in Postgres (§0.8): a copy would always show the mirroring run holding it.

Each table is mirrored on its own: a failing one is reported and the others still are, so the copy is never frozen
by one table. After each load the table's row count is read back from BigQuery (a free metadata call) and must
equal the store's, so an empty table really was emptied. There is no snapshot across tables, and Postgres stays the
only store anything writes to.
"""

from __future__ import annotations

import io
import json
from typing import Any

from sqlalchemy import BigInteger, Boolean, Column, Integer, String, Text

from ..config import BigQueryCfg
from .metadata_store import MetadataStore
from .schemas import TABLES

NOT_MIRRORED = ("run_lock",)
# exact types: Text is a String and BigInteger an Integer to SQLAlchemy, and a type missing here must fail, not guess
BQ_TYPES: dict[type, str] = {
    String: "STRING",
    Text: "STRING",
    Integer: "INT64",
    BigInteger: "INT64",
    Boolean: "BOOL",
}


class MirrorError(RuntimeError):
    """Some tables weren't mirrored (``failures``: table → why; ``loaded``: the ones that were). ``retryable`` when
    BigQuery raised for any of them: the mirror is an idempotent full replace, so the task retries (exit 1);
    otherwise BigQuery holds other rows than the store, which a retry won't change (exit 2)."""

    def __init__(self, failures: dict[str, str], loaded: dict[str, int], retryable: bool) -> None:
        self.failures, self.loaded, self.retryable = failures, loaded, retryable
        super().__init__("; ".join(f"{table}: {why}" for table, why in failures.items()))


def bq_type(column: Column[Any]) -> str:
    try:
        return BQ_TYPES[type(column.type)]
    except KeyError:
        raise ValueError(
            f"{column.table.name}.{column.name}: no BigQuery type for {column.type!r}"
        ) from None


def mirror_metadata(meta: MetadataStore, cfg: BigQueryCfg, client: Any) -> dict[str, int]:
    """Replace each mirrored table in ``cfg.meta_dataset`` with the store's rows; return the rows per table, or raise
    ``MirrorError`` naming every table that failed once all were tried."""
    from google.api_core.exceptions import GoogleAPIError
    from google.cloud import bigquery

    loaded: dict[str, int] = {}
    failures: dict[str, str] = {}
    retryable = False
    for name, table in TABLES.items():
        if name in NOT_MIRRORED:
            continue
        schema = [bigquery.SchemaField(c.name, bq_type(c), mode="NULLABLE") for c in table.columns]
        rows = meta.read(name)
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
            schema=schema,
        )
        destination = f"{cfg.project}.{cfg.meta_dataset}.{name}"
        try:
            client.load_table_from_file(
                io.BytesIO(data), destination, job_config=job_config, location=cfg.location
            ).result()
            held = client.get_table(destination).num_rows
        except GoogleAPIError as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
            retryable = True
            continue
        if held != len(rows):
            failures[name] = f"BigQuery holds {held} rows, the store {len(rows)}"
            continue
        loaded[name] = len(rows)
    if failures:
        raise MirrorError(failures, loaded, retryable)
    return loaded
