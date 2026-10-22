"""The whole bronze node: guards, ordering, flags, the every-file check, the manifest and the lock."""

import json
import re
from datetime import UTC, datetime

import pytest

from precursorintelligence.common.io.run_lock import RunLock, RunLockHeldError, RunLockNotHeldError
from precursorintelligence.harmonize.flags import EVENT_TABLES
from precursorintelligence.harmonize.pipeline import run_bronze
from precursorintelligence.ingestion.pipeline import EXIT_OK, EXIT_VALIDATION, LOCK_NAME

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
ROWS = [("015009", "Oak Manor"), ("15E123", "Pine Court")]
OTHER = [("015009", "Oak Manor"), ("15E123", "Changed")]


def statuses(meta, table, vintage):
    rows = meta.read("bronze_files", {"table_name": table, "vintage_month": vintage})
    return [r["status"] for r in sorted(rows, key=lambda r: r["id"])]


def run(ctx, **kw):
    return run_bronze(ctx, now=NOW, **kw)


def test_every_selected_file_is_loaded_in_vintage_order(ctx, meta, stage):
    for vintage in ("2025-08", "2025-06", "2025-07"):
        stage("provider_info", vintage)
    result = run(ctx)

    assert result.exit_code == EXIT_OK
    assert [(r.vintage, r.status) for r in result.files] == [
        ("2025-06", "loaded"),
        ("2025-07", "loaded"),
        ("2025-08", "loaded"),
    ]


def test_a_rerun_skips_everything_and_writes_no_new_rows(ctx, meta, stage):
    stage("provider_info", "2025-06")
    run(ctx)
    again = run(ctx)

    assert again.exit_code == EXIT_OK and [r.status for r in again.files] == ["skipped"]
    assert statuses(meta, "provider_info", "2025-06") == ["loaded"]


def test_rows_that_are_not_selected_are_ignored(ctx, meta, stage):
    stage("provider_info", "2025-06")
    stage("provider_info", "2025-06", status="duplicate_vintage", zip_sha="b" * 64, member="x.csv")
    result = run(ctx)
    assert result.exit_code == EXIT_OK and len(result.files) == 1


def test_a_pending_inventory_row_refuses_to_start(ctx, meta, stage, sink):
    stage("provider_info", "2025-06")
    stage("penalties", "2025-06", status="pending")
    result = run(ctx)

    assert result.exit_code == EXIT_VALIDATION and "pending" in result.failures[0]
    assert result.files == [] and sink.list_partitions("provider_info") == []


def test_two_selected_files_for_one_table_and_vintage_refuse_to_start(ctx, meta, stage, sink):
    stage("provider_info", "2025-06")
    stage("provider_info", "2025-06", zip_sha="b" * 64, member="other.csv")
    result = run(ctx)

    assert result.exit_code == EXIT_VALIDATION and "provider_info" in result.failures[0]
    assert sink.list_partitions("provider_info") == []


def test_one_bad_file_does_not_stop_the_others_but_fails_the_run(ctx, meta, stage, sink):
    stage("provider_info", "2025-06")
    stage("penalties", "2025-06", row_count=99)  # wrong row count: fails closed
    stage("survey_dates", "2025-06")
    result = run(ctx)

    assert result.exit_code == EXIT_VALIDATION
    assert {r.table: r.status for r in result.files} == {
        "penalties": "failed",
        "provider_info": "loaded",
        "survey_dates": "loaded",
    }
    assert sink.list_partitions("penalties") == []
    assert any("penalties" in f and "2025-06" in f for f in result.failures)
    (step,) = [r for r in meta.read("run_log") if r["task"] == "bronze"]
    assert step["status"] == "failed" and json.loads(step["detail"])["n_failures"] == 1


def test_a_failed_file_loads_on_the_next_run_after_the_fix(ctx, meta, stage):
    stage("penalties", "2025-06", row_count=99)
    run(ctx)
    meta.delete_where("inventory", {"table_name": "penalties"})
    stage("penalties", "2025-06")
    result = run(ctx)

    assert result.exit_code == EXIT_OK
    assert statuses(meta, "penalties", "2025-06") == ["failed", "loaded"]


def test_an_unchanged_release_is_flagged_stale_even_when_an_earlier_vintage_is_republished(
    ctx, meta, stage
):
    for table in EVENT_TABLES:
        stage(table, "2025-07")
        stage(table, "2025-09", zip_sha="b" * 64)
    result = run(ctx)
    flags = {(r["vintage_month"], r["flag"]): r["value"] for r in meta.read("vintage_flags")}

    assert flags[("2025-09", "stale_refresh")] is True
    assert flags[("2025-07", "stale_refresh")] is False
    assert result.stale_vintages == ["2025-09"]

    # CMS republishes 2025-07 with different content: 2025-09 now differs from it.
    meta.delete_where("inventory", {"vintage_month": "2025-07"})
    for table in EVENT_TABLES:
        stage(table, "2025-07", rows=OTHER, zip_sha="c" * 64)
    run(ctx)
    flags = {(r["vintage_month"], r["flag"]): r["value"] for r in meta.read("vintage_flags")}
    assert flags[("2025-09", "stale_refresh")] is False


def test_the_manifest_is_written_with_the_run_and_never_overwritten(ctx, store, meta, stage):
    stage("provider_info", "2025-06", blank_tail=True)
    first = run(ctx)
    second = run(ctx)

    assert re.fullmatch(
        r"metadata/bronze/run_date=\d{4}-\d{2}-\d{2}/bronze_run_R1_\d{8}T\d{12}Z\.json",
        first.manifest_key,
    )
    assert first.manifest_key != second.manifest_key
    manifest = json.loads(store.open_read(first.manifest_key).read())
    assert manifest["run_id"] == "R1" and manifest["exit_code"] == EXIT_OK
    assert manifest["counts"] == {"loaded": 1}
    assert manifest["files"][0] | {} == {
        "table": "provider_info",
        "vintage": "2025-06",
        "status": "loaded",
        "detail": "",
    }
    assert manifest["every_selected_file_has_valid_partition"] == {"passed": True, "missing": []}


def test_the_manifest_of_a_failed_run_names_the_failure(ctx, store, stage):
    stage("penalties", "2025-06", row_count=99)
    result = run(ctx)
    manifest = json.loads(store.open_read(result.manifest_key).read())

    assert manifest["exit_code"] == EXIT_VALIDATION
    assert manifest["every_selected_file_has_valid_partition"]["missing"] == ["penalties 2025-06"]
    assert manifest["counts"] == {"failed": 1}


def test_a_standalone_run_takes_and_releases_the_lock(ctx, meta, stage):
    stage("provider_info", "2025-06")
    run(ctx)
    assert RunLock(meta, LOCK_NAME).holder() is None


def test_a_standalone_run_refuses_while_another_run_holds_the_lock(ctx, meta, stage):
    stage("provider_info", "2025-06")
    RunLock(meta, LOCK_NAME).acquire("OTHER")
    with pytest.raises(RunLockHeldError):
        run(ctx)


def test_a_dag_task_requires_its_run_to_hold_the_lock_and_leaves_it_held(ctx, meta, stage):
    stage("provider_info", "2025-06")
    with pytest.raises(RunLockNotHeldError):
        run(ctx, own_lock=False)

    RunLock(meta, LOCK_NAME).acquire("R1")
    assert run(ctx, own_lock=False).exit_code == EXIT_OK
    assert RunLock(meta, LOCK_NAME).holder()["run_id"] == "R1"
