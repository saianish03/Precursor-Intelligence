"""Run context: the only object tasks receive. Holds config, the three storage ports and the HTTP client."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .config import Settings
from ..ingestion.http import SafeHttpClient
from .io.metadata_store import MetadataStore
from .io.object_store import ObjectStore
from .io.registry import build_metadata_store, build_object_store, build_table_sink
from .io.table_sink import TableSink


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


@dataclass
class Context:
    settings: Settings
    object_store: ObjectStore
    metadata: MetadataStore
    table_sink: TableSink
    http: SafeHttpClient
    run_id: str = field(default_factory=new_run_id)
    started_at: str = field(default_factory=utc_now)

    @property
    def cfg(self):
        return self.settings.sources


def build_context(settings: Settings, *, run_id: str | None = None, http: SafeHttpClient | None = None) -> Context:
    store = build_object_store(settings)
    meta = build_metadata_store(settings)
    meta.ensure_schema()
    s = settings.sources
    return Context(
        settings=settings,
        object_store=store,
        metadata=meta,
        table_sink=build_table_sink(settings, store),
        http=http or SafeHttpClient(s.source, s.http, s.security),
        run_id=run_id or new_run_id(),
    )
