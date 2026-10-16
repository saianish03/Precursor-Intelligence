"""Run lock (D8): one run at a time across processes; re-entrant for tasks of the same run.

Runs on SQLite, and also on Postgres when PI_TEST_POSTGRES_URL points at a throwaway database
(its metadata tables are dropped and recreated)."""

import os
import threading
from datetime import UTC, datetime, timedelta

import pytest

from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore
from precursorintelligence.common.io.run_lock import RunLock, RunLockHeldError
from precursorintelligence.common.io.schemas import metadata

PG_URL = os.environ.get("PI_TEST_POSTGRES_URL")


@pytest.fixture(params=["file", "postgres"])
def db_url(request, tmp_path):
    if request.param == "file":
        return f"sqlite:///{tmp_path / 'm.db'}"
    if not PG_URL:
        pytest.skip("set PI_TEST_POSTGRES_URL to run against Postgres")
    metadata.drop_all(SqlAlchemyMetadataStore(PG_URL).engine)
    return PG_URL


def _store(url):
    m = SqlAlchemyMetadataStore(url)
    m.ensure_schema()
    return m


def test_lock_is_exclusive_between_runs_and_reentrant_within_a_run(db_url):
    a = RunLock(_store(db_url), "ingestion")
    b = RunLock(_store(db_url), "ingestion")  # a second process: its own engine/connection pool
    assert a.acquire("run-1") is True
    assert b.acquire("run-1") is False  # another task of the same DAG run gets in
    with pytest.raises(RunLockHeldError, match="held by run run-1"):
        b.acquire("run-2")
    with pytest.raises(RunLockHeldError):
        b.release("run-2")  # only the holder releases
    assert b.release("run-1") is True
    assert a.holder() is None
    assert a.release("run-1") is False  # releasing a free lock is a no-op
    assert b.acquire("run-2") is True


def test_force_release_breaks_a_lock_left_by_a_killed_run(db_url):
    lock = RunLock(_store(db_url), "ingestion")
    lock.acquire("dead-run")
    assert lock.release("operator", force=True) is True
    assert lock.acquire("next-run") is True


def test_lock_names_are_independent(db_url):
    m = _store(db_url)
    assert RunLock(m, "ingestion").acquire("run-1")
    assert RunLock(m, "harmonize").acquire("run-2")


def test_concurrent_acquire_has_exactly_one_winner(db_url):
    stores = [_store(db_url) for _ in range(8)]
    barrier = threading.Barrier(len(stores))
    outcomes: list[str] = []

    def contend(i):
        barrier.wait()
        try:
            RunLock(stores[i], "ingestion").acquire(f"run-{i}")
            outcomes.append("won")
        except RunLockHeldError:
            outcomes.append("blocked")

    threads = [threading.Thread(target=contend, args=(i,)) for i in range(len(stores))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["blocked"] * 7 + ["won"]


def test_insert_if_absent(db_url):
    m = _store(db_url)
    row = {"lock_name": "x", "run_id": "r"}
    assert m.insert_if_absent("run_lock", row) is True
    assert m.insert_if_absent("run_lock", {**row, "run_id": "other"}) is False
    assert m.read("run_lock")[0]["run_id"] == "r"


def _plant(m, run_id, acquired_at):
    m.upsert("run_lock", [{"lock_name": "ingestion", "run_id": run_id, "acquired_at": acquired_at}])


def test_stale_lock_is_taken_over_only_after_the_threshold(db_url):
    m = _store(db_url)
    lock = RunLock(m, "ingestion")
    young = (datetime.now(UTC) - timedelta(hours=1)).isoformat(timespec="seconds")
    _plant(m, "busy-run", young)
    with pytest.raises(RunLockHeldError):
        lock.acquire("run-2", stale_after=timedelta(hours=12))
    with pytest.raises(RunLockHeldError):
        lock.acquire("run-2")  # no threshold: never taken over

    m.delete_where("run_lock", {"lock_name": "ingestion"})
    _plant(m, "killed-run", "2026-01-01T00:00:00+00:00")
    assert lock.acquire("run-2", stale_after=timedelta(hours=12)) is True
    held = lock.holder()
    assert held["run_id"] == "run-2" and "took over from stale run killed-run" in held["holder"]
