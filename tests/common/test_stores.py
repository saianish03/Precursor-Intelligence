"""Port conformance: the same assertions run against every adapter, proving they are interchangeable."""

import uuid

import polars as pl
import pytest

from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore
from precursorintelligence.common.io.object_store import (
    FsspecObjectStore,
    LocalObjectStore,
    ObjectExistsError,
    ObjectStoreError,
    copy_object,
    sha256_of,
)
from precursorintelligence.common.io.table_sink import ParquetTableSink


@pytest.fixture(params=["local", "memory"])
def store(request, tmp_path):
    if request.param == "local":
        return LocalObjectStore(tmp_path / "root")
    return FsspecObjectStore(f"memory://conf-{uuid.uuid4().hex}")


def test_put_exists_stat_read(store):
    res = store.put_stream("landing/a/b.zip", [b"PK\x03\x04", b"rest"])
    assert res.size == 8 and len(res.sha256) == 64
    assert store.exists("landing/a/b.zip") and not store.exists("landing/a/c.zip")
    assert store.stat("landing/a/b.zip").size == 8
    with store.open_read("landing/a/b.zip") as f:
        assert f.read() == b"PK\x03\x04rest"
    assert sha256_of(store, "landing/a/b.zip") == res.sha256
    assert "landing/a/b.zip" in store.list("landing")


def test_no_silent_overwrite(store):
    store.put_bytes("landing/x.bin", b"1")
    with pytest.raises(ObjectExistsError):
        store.put_bytes("landing/x.bin", b"2")
    store.put_bytes("reports/latest/x.md", b"1")
    store.put_bytes("reports/latest/x.md", b"2", overwrite=True)


@pytest.mark.parametrize("key", ["/abs", "a/../b", "a//b", "a\\b", ""])
def test_key_validation(store, key):
    with pytest.raises(ObjectStoreError):
        store.put_bytes(key, b"x")


def test_move_and_restricted_delete(store):
    store.put_bytes("quarantine/r1/x.zip", b"data")
    store.move("quarantine/r1/x.zip", "landing/z/x.zip")
    assert store.exists("landing/z/x.zip") and not store.exists("quarantine/r1/x.zip")
    with pytest.raises(ObjectStoreError, match="refusing to delete"):
        store.remove("landing/z/x.zip")
    store.put_bytes("quarantine/r1/y.zip", b"d")
    store.remove("quarantine/r1/y.zip")
    assert not store.exists("quarantine/r1/y.zip")


def test_failed_stream_leaves_nothing(store):
    def gen():
        yield b"part"
        raise RuntimeError("network died")
    with pytest.raises(RuntimeError):
        store.put_stream("landing/partial.zip", gen())
    assert not store.exists("landing/partial.zip")


def test_copy_between_backends(tmp_path):
    a = LocalObjectStore(tmp_path / "a")
    b = FsspecObjectStore(f"memory://copy-{uuid.uuid4().hex}")
    a.put_bytes("landing/k.zip", b"abc")
    assert copy_object(a, "landing/k.zip", b, "landing/k.zip").sha256 == sha256_of(a, "landing/k.zip")


def test_parquet_table_sink_on_any_store(store):
    sink = ParquetTableSink(store, prefix="bronze")
    sink.write_partition("provider_info", {"vintage": "2021-01"}, pl.DataFrame({"ccn": ["015009"]}))
    assert sink.list_partitions("provider_info") == [{"vintage": "2021-01"}]
    assert sink.read("provider_info")["ccn"].to_list() == ["015009"]


@pytest.fixture(params=["memory", "file"])
def meta(request, tmp_path):
    url = "sqlite:///:memory:" if request.param == "memory" else f"sqlite:///{tmp_path / 'm.db'}"
    m = SqlAlchemyMetadataStore(url)
    m.ensure_schema()
    return m


def test_metadata_append_upsert_read(meta):
    meta.upsert("catalog", [{"entry_id": "theme:2021-01-27:a.zip", "size_bytes": 1, "active": True}])
    meta.upsert("catalog", [{"entry_id": "theme:2021-01-27:a.zip", "size_bytes": 2, "active": True}])
    rows = meta.read("catalog")
    assert len(rows) == 1 and rows[0]["size_bytes"] == 2
    meta.append("download_log", [{"entry_id": "e", "status": "downloaded"}, {"entry_id": "e", "status": "skipped"}])
    assert [r["status"] for r in meta.read("download_log", {"entry_id": "e"})] == ["downloaded", "skipped"]
    assert meta.read("download_log", {"status": ["skipped"]})[0]["status"] == "skipped"
    assert meta.delete_where("download_log", {"status": "skipped"}) == 1
