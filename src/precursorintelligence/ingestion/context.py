"""Run context: the only object tasks receive. Holds config, the three storage ports and the HTTP client."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..common.config import Settings, SourcesConfig
from ..common.io.metadata_store import MetadataStore
from ..common.io.object_store import ObjectStore
from ..common.io.registry import build_metadata_store, build_object_store, build_table_sink
from ..common.io.table_sink import TableSink
from .http import SafeHttpClient


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def new_run_id() -> str:
    return utc_stamp() + "-" + uuid.uuid4().hex[:6]


# Safe in object keys and log fields, and fits the run_id columns (64). Accepts our own ids and Airflow's
# {{ run_id }}, e.g. scheduled__2026-10-13T00:00:00+00:00.
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,63}")


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            f"invalid run id {run_id!r}: 1-64 characters from A-Z a-z 0-9 _ . : + -, "
            "starting with a letter or digit"
        )
    return run_id


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
    def cfg(self) -> SourcesConfig:
        return self.settings.sources


def build_context(
    settings: Settings, *, run_id: str | None = None, http: SafeHttpClient | None = None
) -> Context:
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
        run_id=validate_run_id(run_id) if run_id else new_run_id(),
    )
