"""Step 1 split into Airflow-style tasks: one process per task, a shared run id, a run lock held across
tasks, per-zip mapped tasks from the pending list, retries with the same run id, and exit codes."""

import json

import pytest
from cms_fakes import make_zip
from test_pipeline_offline import _inv, build_archive

from precursorintelligence.common.io.run_lock import RunLock, RunLockHeldError, RunLockNotHeldError
from precursorintelligence.ingestion import pipeline
from precursorintelligence.ingestion.catalog import select
from precursorintelligence.ingestion.inventory import landing_backlog, pending_entries
from precursorintelligence.ingestion.pipeline import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_VALIDATION,
    LOCK_NAME,
    run_step1,
)
from precursorintelligence.ingestion.resolve import selection_key

RUN = "scheduled__2026-10-13T00:00:00+00:00"  # an Airflow run id


def _dag_run(fake_cms, make_ctx, root, run_id=RUN):
    """The task sequence of the reference DAG; every task builds its own Context (= its own container)."""
    task = lambda: make_ctx(fake_cms, root=root, run_id=run_id)  # noqa: E731
    assert RunLock(task().metadata, LOCK_NAME).acquire(run_id)
    res = run_step1(task(), steps=("discover", "download"), own_lock=False)
    assert res.exit_code == EXIT_OK
    ctx = task()
    todo = [e.entry_id for e in pending_entries(ctx, select(ctx))]
    for entry_id in todo:  # mapped tasks
        mapped = run_step1(
            task(), steps=("extract", "inventory"), entry_ids=[entry_id], own_lock=False
        )
        assert (mapped.selected, mapped.inventoried, mapped.exit_code) == (1, 1, EXIT_OK)
    final = run_step1(task(), steps=("resolve", "report"), own_lock=False)
    assert final.exit_code == EXIT_OK
    return task, todo


def test_split_tasks_match_a_single_run(fake_cms, make_ctx, tmp_path):
    build_archive(fake_cms)
    task, todo = _dag_run(fake_cms, make_ctx, tmp_path / "dag")
    assert len(todo) == 5
    one_shot = make_ctx(fake_cms, root=tmp_path / "single")
    run_step1(one_shot)
    assert _inv(task()) == _inv(one_shot)

    ctx = task()
    assert RunLock(ctx.metadata, LOCK_NAME).holder()["run_id"] == RUN  # tasks never release it
    assert {r["run_id"] for r in ctx.metadata.read("run_log")} == {RUN}
    assert ctx.object_store.exists(f"reports/step1/run={RUN}/01_inventory_summary.md")
    assert RunLock(ctx.metadata, LOCK_NAME).release(RUN)


def test_quiet_day_maps_no_tasks(fake_cms, make_ctx, tmp_path):
    build_archive(fake_cms)
    _dag_run(fake_cms, make_ctx, tmp_path, run_id="run-1")
    RunLock(make_ctx(fake_cms, root=tmp_path).metadata, LOCK_NAME).release("run-1")
    _, todo = _dag_run(fake_cms, make_ctx, tmp_path, run_id="run-2")
    assert todo == []


def test_new_zip_is_the_only_pending_entry(fake_cms, make_ctx, tmp_path):
    build_archive(fake_cms)
    _dag_run(fake_cms, make_ctx, tmp_path, run_id="run-1")
    RunLock(make_ctx(fake_cms, root=tmp_path).metadata, LOCK_NAME).release("run-1")
    fake_cms.add("2021-05-27", make_zip("2021-04", "Apr2021"))
    _, todo = _dag_run(fake_cms, make_ctx, tmp_path, run_id="run-2")
    assert todo == ["theme:2021-05-27:nursing-homes_2021-05-27.zip"]


def test_retrying_a_task_with_the_same_run_id_is_safe(fake_cms, make_ctx, tmp_path):
    build_archive(fake_cms)
    task, _ = _dag_run(fake_cms, make_ctx, tmp_path)
    before = _inv(task())
    for steps in (("discover", "download"), ("extract", "inventory"), ("resolve", "report")):
        assert run_step1(task(), steps=steps, own_lock=False).exit_code == EXIT_OK
    assert _inv(task()) == before


def test_another_run_is_blocked_while_the_lock_is_held(fake_cms, make_ctx, tmp_path):
    build_archive(fake_cms)
    holder = make_ctx(fake_cms, root=tmp_path, run_id="dag-run")
    RunLock(holder.metadata, LOCK_NAME).acquire("dag-run")
    with pytest.raises(RunLockHeldError):
        run_step1(make_ctx(fake_cms, root=tmp_path, run_id="manual-run"))
    assert not fake_cms.requests, "a blocked run must not touch CMS"


def test_standalone_run_releases_its_lock(fake_cms, make_ctx, tmp_path):
    build_archive(fake_cms)
    ctx = make_ctx(fake_cms, root=tmp_path)
    run_step1(ctx)
    assert RunLock(ctx.metadata, LOCK_NAME).holder() is None


def test_dag_task_requires_its_run_to_hold_the_lock(fake_cms, make_ctx, tmp_path):
    build_archive(fake_cms)
    ctx = make_ctx(fake_cms, root=tmp_path, run_id="dag-run")
    with pytest.raises(RunLockNotHeldError, match="is free"):
        run_step1(ctx, steps=("discover",), own_lock=False)
    RunLock(ctx.metadata, LOCK_NAME).acquire("other-run")
    with pytest.raises(RunLockNotHeldError, match="held by run other-run"):
        run_step1(ctx, steps=("discover",), own_lock=False)
    assert not fake_cms.requests


def test_unknown_entry_id_fails(fake_cms, make_ctx):
    build_archive(fake_cms)
    ctx = make_ctx(fake_cms)
    run_step1(ctx, steps=("discover",))
    with pytest.raises(ValueError, match="not in-scope"):
        run_step1(ctx, steps=("extract",), entry_ids=["theme:1999-01-01:nope.zip"])


def test_quarantined_zip_exits_with_validation_code(fake_cms, make_ctx):
    bad = bytearray(make_zip("2021-01", "Jan2021"))
    bad[60] ^= 0xFF
    fake_cms.add("2021-01-27", bytes(bad))
    assert run_step1(make_ctx(fake_cms)).exit_code == EXIT_VALIDATION


def test_task_error_wins_over_validation_failures(fake_cms, make_ctx, monkeypatch):
    build_archive(fake_cms)

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "extract_entry", broken)
    res = run_step1(make_ctx(fake_cms))
    assert res.failures and res.exit_code == EXIT_ERROR


def test_failed_extract_keeps_the_zip_pending(fake_cms, make_ctx, tmp_path, monkeypatch):
    fake_cms.add("2021-01-27", make_zip("2021-01", "Jan2021"))
    ctx = make_ctx(fake_cms, root=tmp_path)
    run_step1(ctx, steps=("discover", "download"))
    [entry_id] = [e.entry_id for e in pending_entries(ctx, select(ctx))]

    def broken(*args, **kwargs):
        raise OSError("GCS 503")

    monkeypatch.setattr(pipeline, "extract_entry", broken)
    res = run_step1(ctx, steps=("extract", "inventory"), entry_ids=[entry_id])
    assert res.exit_code == EXIT_ERROR and res.inventoried == 0
    assert [e.entry_id for e in pending_entries(ctx, select(ctx))] == [entry_id]

    monkeypatch.undo()
    assert run_step1(ctx, steps=("extract", "inventory"), entry_ids=[entry_id]).inventoried == 1
    assert pending_entries(ctx, select(ctx)) == []


def test_inventory_without_extracted_files_stays_pending(fake_cms, make_ctx):
    """Inventory rows of in-scope CSVs without an extracted object (e.g. after an older failed extract)
    must not count as done."""
    fake_cms.add("2021-01-27", make_zip("2021-01", "Jan2021"))
    ctx = make_ctx(fake_cms)
    run_step1(ctx, steps=("discover", "download", "inventory"))  # inventory without extract
    assert len(pending_entries(ctx, select(ctx))) == 1


def test_failed_download_stays_in_the_backlog_after_discover(fake_cms, make_ctx):
    """discover stores the new catalog before downloading; a failed download must stay visible to
    check_release (spec §2.0) until it succeeds."""
    rel = fake_cms.add("2021-01-27", make_zip("2021-01", "Jan2021"))
    ctx = make_ctx(fake_cms)
    run_step1(ctx)
    assert landing_backlog(ctx, select(ctx)).empty

    fake_cms.replace(rel, make_zip("2021-01", "Jan2021", extra={"readme.txt": b"v2"}))
    bad = bytearray(fake_cms.files[rel])
    bad[60] ^= 0xFF  # the re-published bytes arrive corrupt -> quarantined
    fake_cms.files[rel] = bytes(bad)
    assert run_step1(ctx).exit_code == EXIT_VALIDATION
    backlog = landing_backlog(ctx, select(ctx))
    assert backlog.downloads == ["theme:2021-01-27:nursing-homes_2021-01-27.zip"]

    fake_cms.replace(rel, make_zip("2021-01", "Jan2021", extra={"readme.txt": b"v2"}))
    assert run_step1(ctx).exit_code == EXIT_OK
    assert landing_backlog(ctx, select(ctx)).empty


def test_backlog_reports_unextracted_zips_and_pending_rows(fake_cms, make_ctx):
    fake_cms.add("2021-01-27", make_zip("2021-01", "Jan2021"))
    ctx = make_ctx(fake_cms)
    run_step1(ctx, steps=("discover",))
    assert len(landing_backlog(ctx, select(ctx)).downloads) == 1  # advertised, never downloaded
    run_step1(ctx, steps=("download",))
    b = landing_backlog(ctx, select(ctx))
    assert (b.downloads, len(b.extractions), b.pending_inventory_rows) == ([], 1, 0)
    run_step1(ctx, steps=("extract", "inventory"))
    b = landing_backlog(ctx, select(ctx))
    assert b.extractions == [] and b.pending_inventory_rows > 0  # resolve has not run
    run_step1(ctx, steps=("resolve",))
    assert landing_backlog(ctx, select(ctx)).empty


def test_standalone_run_takes_over_a_stale_lock(fake_cms, make_ctx):
    fake_cms.add("2021-01-27", make_zip("2021-01", "Jan2021"))
    ctx = make_ctx(fake_cms)
    ctx.metadata.upsert(
        "run_lock",
        [{"lock_name": LOCK_NAME, "run_id": "killed", "acquired_at": "2026-01-01T00:00:00+00:00"}],
    )
    assert run_step1(ctx).exit_code == EXIT_OK
    assert RunLock(ctx.metadata, LOCK_NAME).holder() is None


def test_report_writes_a_run_manifest_and_retries_add_one(fake_cms, make_ctx, tmp_path):
    build_archive(fake_cms)
    task, _ = _dag_run(fake_cms, make_ctx, tmp_path)
    ctx = task()
    [key] = ctx.object_store.list("metadata/ingestion")
    assert key.startswith("metadata/ingestion/run_date=") and f"ingestion_run_{RUN}_" in key
    m = json.loads(ctx.object_store.open_read(key).read())
    selected = ctx.metadata.read("inventory", {"status": "selected"})
    assert m["run_id"] == RUN and m["latest_vintage"] == "2021-03"
    assert m["counts"]["selected_files"] == len(selected) == len(m["selected"])
    assert m["counts"]["downloads_this_run"] == {"downloaded": 5}
    by_file = {(r["table_name"], r["vintage_month"]): r["zip_sha256"] for r in selected}
    assert {
        (s["table_name"], s["vintage_month"]): s["zip_sha256"] for s in m["selected"]
    } == by_file
    assert m["gaps"] and len(m["sources_config_sha256"]) == 64

    run_step1(task(), steps=("report",), own_lock=False)  # a retry of the same run
    assert len(ctx.object_store.list("metadata/ingestion")) == 2


def test_duplicate_ranking_ties_are_deterministic():
    rows = [
        {"zip_date": "2021-02-27", "zip_type": "theme", "zip_name": n, "member_path": "x.csv"}
        for n in ("b.zip", "a.zip", "c.zip")
    ]
    rows.append({**rows[0], "zip_type": "current", "zip_name": "z.zip"})
    rows.append({**rows[0], "zip_date": "2021-01-27", "zip_name": "zz.zip"})
    for order in (rows, rows[::-1]):
        assert sorted(order, key=selection_key, reverse=True)[0]["zip_name"] == "c.zip"
