"""`bronze --verify`: a read-only report of what bronze holds, to check a real run against the spec's numbers."""

from datetime import UTC, datetime

import pytest
from conftest import csv_text

from precursorintelligence.common.io.run_lock import RunLock
from precursorintelligence.harmonize.flags import EVENT_TABLES
from precursorintelligence.harmonize.pipeline import run_bronze
from precursorintelligence.harmonize.verify import verify_bronze
from precursorintelligence.ingestion.pipeline import LOCK_NAME

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)


@pytest.fixture
def loaded(ctx, stage, put_landing, meta):
    """Three vintages: 2025-07 and 2025-09 identical (stale), the first one cp1252 with a blank last row."""
    old = csv_text([("015009", "St. Mary\x92s")], blank_tail=True, processing="2025-07-01").encode(
        "latin-1"
    )
    for table in EVENT_TABLES:
        inv = put_landing(
            old, table=table, vintage="2025-07", processing_date="2025-07-01", member=f"{table}.csv"
        )
        meta.append("inventory", [inv | {"status": "selected"}])
        stage(table, "2025-09", rows=[("015009", "St. Mary’s")], zip_sha="b" * 64)
    stage("provider_info", "2025-09", zip_sha="b" * 64)
    run_bronze(ctx, now=NOW)
    return ctx


def test_the_report_counts_files_encodings_blank_rows_and_stale_vintages(loaded):
    report = verify_bronze(loaded)

    assert report.selected_files == 7
    assert report.valid_partitions == report.selected_files
    assert report.encodings == {"cp1252": 3, "utf-8": 4}
    assert report.blank_row_files == {"health_citations": 1, "penalties": 1, "survey_dates": 1}
    assert report.vintages == ("2025-07", "2025-09")
    assert report.stale_refresh_vintages == ["2025-09"]
    assert report.revised == [] and report.pending_inventory_rows == 0
    assert report.ok


def test_a_lost_partition_or_a_pending_row_makes_the_report_fail(loaded, meta, drop_object):
    drop_object("bronze/penalties/vintage=2025-09/part-0.parquet")
    meta.append(
        "inventory",
        [
            {
                "entry_id": "e",
                "member_path": "p",
                "table_name": "mds_qm",
                "vintage_month": "2025-09",
                "status": "pending",
            }
        ],
    )
    report = verify_bronze(loaded)

    assert not report.ok
    assert report.missing == ["penalties 2025-09"]
    assert report.pending_inventory_rows == 1
    assert report.valid_partitions == report.selected_files - 1


def test_the_report_reads_only(loaded, store, meta):
    objects, rows = store.list(""), meta.read("bronze_files")
    RunLock(meta, LOCK_NAME).acquire("another-run")
    verify_bronze(loaded)

    assert store.list("") == objects
    assert meta.read("bronze_files") == rows
    assert RunLock(meta, LOCK_NAME).holder()["run_id"] == "another-run"


def test_the_rendered_report_shows_the_numbers_to_compare_with_the_spec(loaded):
    text = verify_bronze(loaded).render()
    for expected in (
        "selected files",
        "valid partitions",
        "cp1252=3 utf-8=4",
        "2025-07 .. 2025-09",
        "stale_refresh",
    ):
        assert expected in text


def test_an_empty_bronze_is_reported_not_crashed(ctx):
    report = verify_bronze(ctx)
    assert report.selected_files == 0 and report.vintages == () and report.ok
    assert "none" in report.render()
