"""Fixtures for the bronze tests: a store (local disk and memory), a metadata DB, a sink, and landing files."""

import uuid

import polars as pl
import pytest

from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore
from precursorintelligence.common.io.object_store import FsspecObjectStore, LocalObjectStore
from precursorintelligence.common.io.table_sink import ParquetTableSink
from precursorintelligence.harmonize.reader import decode_csv, parse_csv_text
from precursorintelligence.ingestion.layout import header_hash, normalize_header

ROWS = [("015009", "Oak Manor"), ("15E123", "Pine Court")]
HEADER = '"CMS Certification Number (CCN)","Provider Name","Processing Date"\r\n'


def csv_text(rows, *, blank_tail=False, processing="2025-06-01"):
    body = "".join(f'"{ccn}","{name}","{processing}"\r\n' for ccn, name in rows)
    return HEADER + body + ("\r\n" if blank_tail else "")


@pytest.fixture(params=["local", "memory"])
def store(request, tmp_path):
    if request.param == "local":
        return LocalObjectStore(tmp_path / "root")
    return FsspecObjectStore(f"memory://bronze-{uuid.uuid4().hex}")


@pytest.fixture
def meta(tmp_path):
    db = SqlAlchemyMetadataStore(f"sqlite:///{tmp_path / 'meta.db'}")
    db.ensure_schema()
    return db


@pytest.fixture
def sink(store):
    return ParquetTableSink(store, prefix="bronze")


@pytest.fixture
def put_landing(store):
    """Write one CSV to landing and return the inventory row Step 1 would have recorded for it."""

    def put(
        content,
        *,
        table="provider_info",
        vintage="2025-06",
        zip_sha="a" * 64,
        processing_date="2025-06-01",
        member="NH_ProviderInfo_Jun2025.csv",
        row_count=None,
        header=None,
    ):
        raw = content if isinstance(content, bytes) else content.encode("utf-8")
        parsed = None if row_count is not None and header else parse_csv_text(decode_csv(raw)[0])
        key = f"landing/cms_nh/extracted/zip_date={vintage}-27/nh_{vintage}/{member}"
        store.put_bytes(key, raw, overwrite=True)
        return {
            "entry_id": f"current:{vintage}-27:nh_{vintage}.zip",
            "member_path": member,
            "zip_name": f"nh_{vintage}.zip",
            "zip_sha256": zip_sha,
            "table_name": table,
            "extracted_key": key,
            "processing_date": processing_date,
            "vintage_month": vintage,
            "row_count": parsed.frame.height + parsed.blank_rows
            if row_count is None
            else row_count,
            "header_hash": header or header_hash(normalize_header(parsed.frame.columns)),
            "status": "selected",
        }

    return put


@pytest.fixture
def read_partition(sink):
    def read(table, vintage) -> pl.DataFrame:
        return sink.read(table, {"vintage": vintage})

    return read


@pytest.fixture
def drop_object(store):
    """Delete an object the way a bucket mishap would; the port itself only deletes scratch prefixes."""

    def drop(key):
        path = store.local_path(key)
        if path is not None:
            path.unlink()
        else:
            store.fs.rm(store._path(key))

    return drop


@pytest.fixture
def ctx(store, meta, sink, tmp_path):
    """A run context over the same store, metadata DB and sink the other fixtures use."""
    import yaml

    from precursorintelligence.common.config import load_settings
    from precursorintelligence.ingestion.context import Context

    env = tmp_path / "env.yaml"
    env.write_text(
        yaml.safe_dump(
            {
                "env": "test-bronze",
                "object_store": {"kind": "local", "root": str(tmp_path / "unused")},
                "metadata_store": {"url": f"sqlite:///{tmp_path / 'unused.db'}"},
            }
        )
    )
    return Context(
        settings=load_settings("dev", env_path=env),
        object_store=store,
        metadata=meta,
        table_sink=sink,
        http=None,  # type: ignore[arg-type]  # bronze never touches the network
        run_id="R1",
    )


@pytest.fixture
def stage(meta, put_landing):
    """Put a CSV in landing and record its (selected) inventory row, as Step 1 would."""

    def put(table, vintage, rows=ROWS, status="selected", zip_sha="a" * 64, **over):
        inv = put_landing(
            csv_text(rows, processing=f"{vintage}-01", blank_tail=over.pop("blank_tail", False)),
            table=table,
            vintage=vintage,
            processing_date=f"{vintage}-01",
            member=over.pop("member", f"NH_{table}_{vintage}.csv"),
            zip_sha=zip_sha,
            **over,
        )
        meta.append("inventory", [inv | {"status": status}])
        return inv

    return put
