"""The evidence the agents read (docs/AGENTS_PLAN.md E1): one ``EvidenceSource`` interface, so the agents never know
where the data lives. ``ParquetEvidenceSource`` reads the files that ``build-evidence`` (E2) writes; a BigQuery
source can follow with the same methods (every query setting ``maximum_bytes_billed``).

The layout, pointer and manifest are in ``evidence.contract``. The pointer must match the manifest's sha256, and
each file its manifest entry, or the read refuses (``EvidenceError``): a half-written month is never served. ``citations`` and
``penalties`` have no ``gold_run_id`` column: they come from the silver of vintage ``v``, not from the gold run, and
the manifest's sha256 ties them to this build.
``scores_manifest_sha256`` is not checked here. The source keeps the pointer it read (``pointer``), so the answer
cache's key (D1) comes from the same read as the evidence; the runner reopens the source when ``_current.json``
changes.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from typing import Protocol

import polars as pl
from pydantic import ValidationError

from ...common.io.object_store import ObjectStore
from ...evidence.contract import (
    BUILD_PATTERN,
    COLUMN_TYPES,
    CURRENT,
    EVIDENCE_MANIFEST,
    TABLE_COLUMNS,
    V_PATTERN,
    CurrentPointer,
    EvidenceManifest,
    Table,
    build_dir,
)

# The columns of each table that may hold nulls: the optional fields of the tools' records. A null anywhere else
# is refused at ``open``, so a record never fails to build inside a tool.
NULLABLE: dict[Table, frozenset[str]] = {
    "facility_scores": frozenset({"state", "chain_id"}),
    "facility_reasons": frozenset({"feature_value"}),
    "citations": frozenset({"correction_date"}),
    "penalties": frozenset(
        {"fine_amount", "payment_denial_start_date", "payment_denial_length_days"}
    ),
    "chain_scores": frozenset({"bed_weighted_mean_p"}),
}


class EvidenceError(RuntimeError):
    """The evidence is missing, inconsistent or does not match its manifest."""


class EvidenceSource(Protocol):
    """Read-only access to one month's evidence."""

    @property
    def pointer(self) -> CurrentPointer:
        """The ``_current.json`` this source was opened from."""
        ...

    @property
    def manifest(self) -> EvidenceManifest: ...

    def rows(self, table: Table, key: str, values: Sequence[str]) -> pl.DataFrame:
        """The rows of ``table`` whose ``key`` column is in ``values`` (all of ``TABLE_COLUMNS[table]``)."""
        ...


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ParquetEvidenceSource:
    """The evidence of the month in ``<prefix>_current.json``, from parquet in an ``ObjectStore`` (GCS or local).

    ``open`` reads and checks the pointer, the manifest and every table, and keeps the tables in memory (they are
    small, and fixed for the month): a bad file is refused at startup, and tool calls never touch the store, so a
    stuck read can only fail the instance's start.
    """

    def __init__(
        self, pointer: CurrentPointer, manifest: EvidenceManifest, tables: dict[Table, pl.DataFrame]
    ) -> None:
        self._pointer = pointer
        self._manifest = manifest
        self._tables = tables

    @classmethod
    def open(cls, store: ObjectStore, prefix: str) -> ParquetEvidenceSource:
        if prefix and not prefix.endswith("/"):
            raise ValueError(f"prefix must end with '/': {prefix!r}")
        pointer_key = f"{prefix}{CURRENT}"
        try:
            pointer = CurrentPointer.model_validate_json(_read(store, pointer_key))
        except ValidationError as exc:
            raise EvidenceError(
                f"{pointer_key} is not a valid pointer: {exc.error_count()} error(s)"
            ) from None
        if not V_PATTERN.match(pointer.v) or not BUILD_PATTERN.match(pointer.build):
            raise EvidenceError(
                f"{pointer_key} names month {pointer.v!r} build {pointer.build!r}: not YYYY-MM and 12 "
                "hex characters"
            )

        directory = build_dir(prefix, pointer.v, pointer.build)
        manifest_key = f"{directory}{EVIDENCE_MANIFEST}"
        raw = _read(store, manifest_key)
        if _sha256(raw) != pointer.evidence_manifest_sha256:
            raise EvidenceError(f"{manifest_key} does not match the sha256 in {pointer_key}")
        try:
            manifest = EvidenceManifest.model_validate_json(raw)
        except ValidationError as exc:
            raise EvidenceError(
                f"{manifest_key} is not a valid manifest: {exc.error_count()} error(s)"
            ) from None
        if manifest.v != pointer.v:
            raise EvidenceError(f"{manifest_key} is for {manifest.v}, the pointer for {pointer.v}")
        missing = sorted(set(TABLE_COLUMNS) - set(manifest.files))
        if missing:
            raise EvidenceError(f"{manifest_key} lacks the tables {missing}")
        for name, key in manifest.files.items():
            expected = f"{directory}{name}.parquet" if name in TABLE_COLUMNS else None
            inside = key.startswith(directory) and "/" not in key[len(directory) :]
            if (
                expected and key != expected
            ) or not inside:  # never another month's or build's file
                raise EvidenceError(f"{manifest_key} points {name} at {key}, outside {directory}")
        tables = {t: _load(store, manifest, t) for t in TABLE_COLUMNS}
        return cls(pointer, manifest, tables)

    @property
    def pointer(self) -> CurrentPointer:
        return self._pointer

    @property
    def manifest(self) -> EvidenceManifest:
        return self._manifest

    def rows(self, table: Table, key: str, values: Sequence[str]) -> pl.DataFrame:
        return self._tables[table].filter(pl.col(key).is_in(list(values)))


def _load(store: ObjectStore, m: EvidenceManifest, table: Table) -> pl.DataFrame:
    key = m.files[table]
    data = _read(store, key)
    if _sha256(data) != m.sha256.get(table):
        raise EvidenceError(f"{key} does not match its sha256 in {EVIDENCE_MANIFEST}")
    df = pl.read_parquet(io.BytesIO(data))
    if df.height != m.rows.get(table):
        raise EvidenceError(f"{key} has {df.height} rows, the manifest says {m.rows.get(table)}")
    missing = [c for c in TABLE_COLUMNS[table] if c not in df.columns]
    if missing:
        raise EvidenceError(f"{key} lacks the columns {missing}")
    wrong = {
        c: str(df.schema[c])
        for c in TABLE_COLUMNS[table]
        if c in COLUMN_TYPES and df.schema[c] not in (COLUMN_TYPES[c], pl.Null)
    }
    if wrong:
        raise EvidenceError(f"{key} has columns of the wrong type: {wrong}")
    nulls = [c for c in TABLE_COLUMNS[table] if c not in NULLABLE[table] and df[c].null_count()]
    if nulls:
        raise EvidenceError(f"{key} has nulls in {nulls}")
    if "gold_run_id" in df.columns and set(df["gold_run_id"].unique().to_list()) - {m.gold_run_id}:
        raise EvidenceError(f"{key} holds rows of another gold run than {m.gold_run_id}")
    if "model_version" in df.columns and set(df["model_version"].unique().to_list()) - {
        m.model_version
    }:
        raise EvidenceError(f"{key} holds rows of another model version than {m.model_version}")
    return df


def _read(store: ObjectStore, key: str) -> bytes:
    if not store.exists(key):
        raise EvidenceError(f"{key} is missing")
    with store.open_read(key) as f:
        return f.read()
