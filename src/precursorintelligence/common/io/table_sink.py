"""TableSink port: where analytical tables (bronze/silver/gold) are written. Used from Step 2 onward.

Defined now so Step 2 can plug in without touching Step 1. Adapters:
  * ParquetTableSink - Hive-partitioned Parquet on any ObjectStore (local disk, S3, GCS).
                       On AWS register the prefix in Glue to query with Athena; on GCP use BigLake.
  * DuckDBTableSink  - a local analytical database file.
  * BigQuery / Redshift adapters are intentionally left as stubs until a cloud project exists.
"""

from __future__ import annotations

import io
import logging
from typing import Protocol, runtime_checkable

import polars as pl

from .object_store import ObjectStore, validate_key

log = logging.getLogger(__name__)


@runtime_checkable
class TableSink(Protocol):
    def write_partition(self, table: str, partition: dict[str, str], df: pl.DataFrame,
                        *, overwrite: bool = False) -> str: ...
    def list_partitions(self, table: str) -> list[dict[str, str]]: ...
    def read(self, table: str, partition: dict[str, str] | None = None) -> pl.DataFrame: ...


def _partition_path(partition: dict[str, str]) -> str:
    return "/".join(f"{k}={v}" for k, v in partition.items())


class ParquetTableSink:
    def __init__(self, store: ObjectStore, prefix: str = "bronze") -> None:
        self.store, self.prefix = store, prefix.strip("/")

    def _key(self, table: str, partition: dict[str, str]) -> str:
        return validate_key(f"{self.prefix}/{table}/{_partition_path(partition)}/part-0.parquet")

    def write_partition(self, table, partition, df, *, overwrite=False) -> str:
        buf = io.BytesIO()
        df.write_parquet(buf, compression="zstd")
        key = self._key(table, partition)
        self.store.put_bytes(key, buf.getvalue(), overwrite=overwrite)
        return self.store.uri_for(key)

    def list_partitions(self, table: str) -> list[dict[str, str]]:
        out = []
        for key in self.store.list(f"{self.prefix}/{table}"):
            parts = key.split("/")[2:-1]
            out.append(dict(p.split("=", 1) for p in parts if "=" in p))
        return out

    def read(self, table, partition=None) -> pl.DataFrame:
        frames = []
        for p in self.list_partitions(table):
            if partition and any(p.get(k) != v for k, v in partition.items()):
                continue
            with self.store.open_read(self._key(table, p)) as f:
                frames.append(pl.read_parquet(f).with_columns([pl.lit(v).alias(k) for k, v in p.items()]))
        return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()


class DuckDBTableSink:
    def __init__(self, path: str) -> None:
        import duckdb

        self.con = duckdb.connect(path)

    def write_partition(self, table, partition, df, *, overwrite=False) -> str:
        df = df.with_columns([pl.lit(v).alias(k) for k, v in partition.items()])
        self.con.register("_incoming", df.to_arrow())
        exists = self.con.execute(
            "select count(*) from information_schema.tables where table_name = ?", [table]).fetchone()[0]
        if not exists:
            self.con.execute(f'create table "{table}" as select * from _incoming limit 0')
        cond = " and ".join(f'"{k}" = ?' for k in partition)
        if self.con.execute(f'select count(*) from "{table}" where {cond}', list(partition.values())).fetchone()[0]:
            if not overwrite:
                raise FileExistsError(f"partition {partition} of {table} already exists")
            self.con.execute(f'delete from "{table}" where {cond}', list(partition.values()))
        self.con.execute(f'insert into "{table}" by name select * from _incoming')
        self.con.unregister("_incoming")
        return f"duckdb:{table}/{_partition_path(partition)}"

    def list_partitions(self, table):  # pragma: no cover - thin wrapper
        raise NotImplementedError("query the DuckDB table directly")

    def read(self, table, partition=None) -> pl.DataFrame:
        return self.con.execute(f'select * from "{table}"').pl()
