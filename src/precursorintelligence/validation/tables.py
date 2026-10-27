"""Read a layer's tables month by month, straight from the object store.

Read-only: no lock, no build step and no freshness check, so it also works on a silver built with older configs.
  silver: ``silver/<table>/vintage=YYYY-MM/part-0.parquet``, one file per month (``partition_key``)
  gold:   ``gold/<table>/part-0.parquet``, every month in one file, split by ``month_column``
"""

from __future__ import annotations

import polars as pl

from ..common.config import ValidationLayer
from ..common.io.object_store import ObjectStore


class TableReader:
    def __init__(self, store: ObjectStore, layer: str, cfg: ValidationLayer, table: str) -> None:
        self.store, self.layer, self.cfg, self.table = store, layer, cfg, table
        self._whole: pl.DataFrame | None = None  # gold: the one file, read once

    def _read(self, key: str) -> pl.DataFrame:
        with self.store.open_read(key) as f:
            return pl.read_parquet(f)

    def _partition_keys(self) -> dict[str, str]:
        """month -> object key of its part file."""
        out: dict[str, str] = {}
        for key in self.store.list(f"{self.layer}/{self.table}/"):
            parts = key.split("/")
            if parts[-1] != "part-0.parquet":
                continue
            for part in parts[2:-1]:
                name, _, value = part.partition("=")
                if name == self.cfg.partition_key:
                    out[value] = key
        return out

    def _whole_table(self) -> pl.DataFrame | None:
        if self._whole is None:
            key = f"{self.layer}/{self.table}/part-0.parquet"
            if not self.store.exists(key):
                return None
            self._whole = self._read(key)
        return self._whole

    def months(self) -> list[str]:
        """The table's months in order; [] when the table is not in the store."""
        if self.cfg.partition_key:
            return sorted(self._partition_keys())
        df = self._whole_table()
        if df is None or self.cfg.month_column not in df.columns:
            return []
        return sorted(str(m) for m in df[self.cfg.month_column].drop_nulls().unique())

    def month(self, month: str) -> pl.DataFrame:
        """One month's rows, with the columns as stored."""
        if self.cfg.partition_key:
            return self._read(self._partition_keys()[month])
        df = self._whole_table()
        if df is None:
            raise KeyError(f"{self.layer}/{self.table} is not in the store")
        col = pl.col(str(self.cfg.month_column))
        return df.filter(col.cast(pl.String) == month)
