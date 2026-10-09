"""Adapter registry: maps the ``kind`` in configs/env/<env>.yaml to a concrete adapter.

Adding a backend = write an adapter class that satisfies the port Protocol, then register it::

    @register_object_store("azure")
    def _azure(cfg, settings): return FsspecObjectStore(cfg.root, storage_options=cfg.storage_options)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import ObjectStoreCfg, Settings, TableSinkCfg
from .metadata_store import MetadataStore, SqlAlchemyMetadataStore
from .object_store import FsspecObjectStore, LocalObjectStore, ObjectStore
from .table_sink import DuckDBTableSink, ParquetTableSink, TableSink

_OBJECT_STORES: dict[str, Callable[[ObjectStoreCfg, Settings], ObjectStore]] = {}
_TABLE_SINKS: dict[str, Callable[[TableSinkCfg, Settings, ObjectStore], TableSink]] = {}


def register_object_store(kind: str) -> Callable[[Any], Any]:
    def deco(factory):
        _OBJECT_STORES[kind] = factory
        return factory
    return deco


def register_table_sink(kind: str) -> Callable[[Any], Any]:
    def deco(factory):
        _TABLE_SINKS[kind] = factory
        return factory
    return deco


@register_object_store("local")
def _local(cfg: ObjectStoreCfg, settings: Settings) -> ObjectStore:
    return LocalObjectStore(cfg.root, immutable=cfg.immutable or "chmod")


@register_object_store("memory")
@register_object_store("s3")
@register_object_store("gcs")
def _fsspec(cfg: ObjectStoreCfg, settings: Settings) -> ObjectStore:
    return FsspecObjectStore(cfg.root, immutable=cfg.immutable, storage_options=cfg.storage_options)


@register_table_sink("parquet")
def _parquet(cfg: TableSinkCfg, settings: Settings, store: ObjectStore) -> TableSink:
    return ParquetTableSink(store, prefix=cfg.prefix)


@register_table_sink("duckdb")
def _duckdb(cfg: TableSinkCfg, settings: Settings, store: ObjectStore) -> TableSink:
    return DuckDBTableSink(str(settings.resolve_path(cfg.options.get("path", "./data/warehouse.duckdb"))))


def build_object_store(settings: Settings) -> ObjectStore:
    cfg = settings.env.object_store
    try:
        return _OBJECT_STORES[cfg.kind](cfg, settings)
    except KeyError:
        raise ValueError(f"unknown object_store.kind {cfg.kind!r}; registered: {sorted(_OBJECT_STORES)}") from None


def build_metadata_store(settings: Settings) -> MetadataStore:
    return SqlAlchemyMetadataStore(settings.env.metadata_store.url)


def build_table_sink(settings: Settings, store: ObjectStore) -> TableSink:
    cfg = settings.env.table_sink
    try:
        return _TABLE_SINKS[cfg.kind](cfg, settings, store)
    except KeyError:
        raise ValueError(f"unknown table_sink.kind {cfg.kind!r}; registered: {sorted(_TABLE_SINKS)}") from None
