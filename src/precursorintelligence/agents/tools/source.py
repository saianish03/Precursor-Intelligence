"""The evidence the agents read (docs/AGENTS_PLAN.md E1): one ``EvidenceSource`` interface, so the agents never know
where the data lives. ``ParquetEvidenceSource`` reads the files that ``build-evidence`` (E2) writes; a BigQuery
source can follow with the same methods (every query setting ``maximum_bytes_billed``).

Layout under ``prefix`` (e.g. ``agents/evidence/``)::

    _current.json                         written last: {"schema_version", "v", "scores_manifest_sha256",
                                                         "evidence_manifest_sha256"}
    v=YYYY-MM/evidence_manifest.json      what the month's files are: v, gold_run_id, model_version, built_at,
                                          and per file its key, row count and sha256
    v=YYYY-MM/<table>.parquet             one file per name in TABLE_COLUMNS
    v=YYYY-MM/<other files>               e.g. the ops evidence of Agent 3 (E2, I1); listed in the manifest, not
                                          read here

The month comes only from ``_current.json``: never from ``status/latest.json`` (its ``v`` can be a month with
nothing scored) and never from listing the prefix. The pointer must match the manifest's sha256, and each file its
manifest entry, or the read refuses (``EvidenceError``): a half-written month is never served. ``citations`` and
``penalties`` have no ``gold_run_id`` column: the manifest's sha256 is what ties them to its gold run.
``scores_manifest_sha256`` is not checked here; it is for the answer cache's key (D1).
"""

from __future__ import annotations

import hashlib
import io
import re
import threading
from collections.abc import Sequence
from typing import Literal, Protocol

import polars as pl
from pydantic import BaseModel, ConfigDict, ValidationError

from ...common.io.object_store import ObjectStore

CURRENT = "_current.json"
EVIDENCE_MANIFEST = "evidence_manifest.json"
V_PATTERN = re.compile(r"^20(2[1-6])-(0[1-9]|1[0-2])$")  # vintages 2021-2026 only

Table = Literal["facility_scores", "facility_reasons", "citations", "penalties", "chain_scores"]

# The columns each table must have; a file may carry more. ``p`` is the calibrated probability of the champion's
# label (``p_<label>`` in scores/); build-evidence renames it so the tools need not know the label.
TABLE_COLUMNS: dict[Table, tuple[str, ...]] = {
    "facility_scores": (
        "ccn",
        "state",
        "chain_id",
        "p",
        "risk_percentile",
        "risk_decile",
        "model_version",
        "gold_run_id",
    ),
    "facility_reasons": (
        "ccn",
        "direction",
        "rank",
        "feature",
        "feature_value",
        "contribution_logodds",
        "label",
        "model_version",
        "gold_run_id",
    ),
    "citations": (
        "ccn",
        "survey_date",
        "ftag",
        "deficiency_category",
        "scope_severity",
        "is_standard",
        "is_complaint",
        "correction_date",
    ),
    "penalties": (
        "ccn",
        "penalty_date",
        "penalty_type",
        "fine_amount",
        "payment_denial_start_date",
        "payment_denial_length_days",
        "dup_seq",
    ),
    "chain_scores": (
        "chain_id",
        "n_facilities",
        "beds",
        "expected_facilities_with_event",
        "p_any_event",
        "bed_weighted_mean_p",
        "max_p",
        "worst_ccn",
        "share_top_decile",
        "model_version",
        "gold_run_id",
    ),
}

# Columns whose type the tools rely on: a CCN read as a number loses its leading zeros and matches nothing.
COLUMN_TYPES: dict[str, type[pl.DataType]] = {
    "ccn": pl.String,
    "worst_ccn": pl.String,
    "chain_id": pl.String,
    "survey_date": pl.Date,
    "correction_date": pl.Date,
    "penalty_date": pl.Date,
    "payment_denial_start_date": pl.Date,
}


class EvidenceError(RuntimeError):
    """The evidence is missing, inconsistent or does not match its manifest."""


class CurrentPointer(BaseModel):
    """``_current.json``: the month the agents serve."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    v: str
    scores_manifest_sha256: str
    evidence_manifest_sha256: str


class EvidenceManifest(BaseModel):
    """``v=YYYY-MM/evidence_manifest.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    v: str
    gold_run_id: str
    model_version: int
    built_at: str
    files: dict[str, str]  # table -> object key
    rows: dict[str, int]
    sha256: dict[str, str]


class EvidenceSource(Protocol):
    """Read-only access to one month's evidence."""

    @property
    def manifest(self) -> EvidenceManifest: ...

    def rows(self, table: Table, key: str, values: Sequence[str]) -> pl.DataFrame:
        """The rows of ``table`` whose ``key`` column is in ``values`` (all of ``TABLE_COLUMNS[table]``)."""
        ...


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ParquetEvidenceSource:
    """The evidence of the month in ``<prefix>_current.json``, from parquet in an ``ObjectStore`` (GCS or local).

    ``open`` reads and checks the pointer and the manifest; each table is read, checked against the manifest and
    kept in memory on first use (they are small, and fixed for the month).
    """

    def __init__(self, store: ObjectStore, manifest: EvidenceManifest) -> None:
        self._store = store
        self._manifest = manifest
        self._tables: dict[Table, pl.DataFrame] = {}
        self._locks = {
            t: threading.Lock() for t in TABLE_COLUMNS
        }  # a slow table never blocks another

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
        if not V_PATTERN.match(pointer.v):
            raise EvidenceError(
                f"{pointer_key} names month {pointer.v!r}, not YYYY-MM in 2021-2026"
            )

        directory = f"{prefix}v={pointer.v}/"
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
            if (expected and key != expected) or not inside:  # never another month's file
                raise EvidenceError(f"{manifest_key} points {name} at {key}, outside {directory}")
        return cls(store, manifest)

    @property
    def manifest(self) -> EvidenceManifest:
        return self._manifest

    def rows(self, table: Table, key: str, values: Sequence[str]) -> pl.DataFrame:
        df = self._table(table)
        return df.filter(pl.col(key).is_in(list(values)))

    def _table(self, table: Table) -> pl.DataFrame:
        with self._locks[table]:
            if table not in self._tables:
                self._tables[table] = self._load(table)
            return self._tables[table]

    def _load(self, table: Table) -> pl.DataFrame:
        m = self._manifest
        key = m.files[table]
        data = _read(self._store, key)
        if _sha256(data) != m.sha256.get(table):
            raise EvidenceError(f"{key} does not match its sha256 in {EVIDENCE_MANIFEST}")
        df = pl.read_parquet(io.BytesIO(data))
        if df.height != m.rows.get(table):
            raise EvidenceError(
                f"{key} has {df.height} rows, the manifest says {m.rows.get(table)}"
            )
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
        if "gold_run_id" in df.columns and set(df["gold_run_id"].unique().to_list()) - {
            m.gold_run_id
        }:
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
