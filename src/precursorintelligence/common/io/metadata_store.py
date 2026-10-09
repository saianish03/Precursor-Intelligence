"""MetadataStore port and its SQLAlchemy adapter.

One adapter covers every SQL database SQLAlchemy supports. Changing the backend is a URL change:
  sqlite:///./data/ops/metadata.db             (local)
  postgresql+psycopg://user@rds-host/db        (AWS RDS / Aurora)
  postgresql+psycopg://user@/db?host=/cloudsql/<instance>  (GCP Cloud SQL)
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import Integer, and_, create_engine, delete, insert, select
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from .schemas import TABLES, metadata

log = logging.getLogger(__name__)


@runtime_checkable
class MetadataStore(Protocol):
    def ensure_schema(self) -> None: ...
    def append(self, table: str, rows: Iterable[Mapping[str, Any]]) -> int: ...
    def upsert(self, table: str, rows: Iterable[Mapping[str, Any]]) -> int: ...
    def read(self, table: str, where: Mapping[str, Any] | None = None) -> list[dict[str, Any]]: ...
    def delete_where(self, table: str, where: Mapping[str, Any]) -> int: ...


class SqlAlchemyMetadataStore:
    """Thread-safe (serialized writes) SQLAlchemy Core implementation."""

    def __init__(self, url: str) -> None:
        kwargs: dict[str, Any] = {"future": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool
            else:
                Path(url.split("sqlite:///", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        self.url = url
        self.engine: Engine = create_engine(url, **kwargs)
        self._lock = threading.RLock()

    @property
    def safe_url(self) -> str:
        return self.engine.url.render_as_string(hide_password=True)

    def ensure_schema(self) -> None:
        with self._lock:
            metadata.create_all(self.engine)

    def _table(self, name: str):
        try:
            return TABLES[name]
        except KeyError:
            raise KeyError(f"unknown metadata table: {name}") from None

    @staticmethod
    def _clean(table, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Uniform rows for executemany: unknown keys dropped, missing columns set to NULL.
        Integer surrogate keys (autoincrement ``id``) are left to the database."""
        auto = {c.name for c in table.primary_key.columns if isinstance(c.type, Integer)}
        cols = [c.name for c in table.c if c.name not in auto]
        return [{c: r.get(c) for c in cols} for r in rows]

    def append(self, table: str, rows: Iterable[Mapping[str, Any]]) -> int:
        t = self._table(table)
        data = self._clean(t, rows)
        if not data:
            return 0
        with self._lock, self.engine.begin() as conn:
            conn.execute(insert(t), data)
        return len(data)

    def upsert(self, table: str, rows: Iterable[Mapping[str, Any]]) -> int:
        """Replace rows that share the primary key (portable delete+insert in one transaction)."""
        t = self._table(table)
        pk = list(t.primary_key.columns)
        data = self._clean(t, rows)
        if not data:
            return 0
        with self._lock, self.engine.begin() as conn:
            for row in data:
                conn.execute(delete(t).where(and_(*[c == row[c.name] for c in pk])))
            conn.execute(insert(t), data)
        return len(data)

    def read(self, table: str, where: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        t = self._table(table)
        stmt = select(t)
        for k, v in (where or {}).items():
            stmt = stmt.where(t.c[k].in_(v) if isinstance(v, (list, tuple, set)) else t.c[k] == v)
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(stmt)]

    def delete_where(self, table: str, where: Mapping[str, Any]) -> int:
        t = self._table(table)
        stmt = delete(t)
        for k, v in where.items():
            stmt = stmt.where(t.c[k] == v)
        with self._lock, self.engine.begin() as conn:
            return conn.execute(stmt).rowcount or 0
