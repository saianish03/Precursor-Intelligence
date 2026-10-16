"""One file at a time: skip, load, reload or fail (docs/DATA_PIPELINE_SPEC.md §3.1 steps 1 to 9)."""

from datetime import UTC, datetime

import polars as pl
from conftest import csv_text

from precursorintelligence.harmonize.bronze import load_file

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
ROWS = [("015009", "Oak Manor"), ("15E123", "Pine Court")]


def load(store, meta, sink, inv, run_id="R1"):
    return load_file(store, meta, sink, inv, run_id=run_id, now=NOW)


def files(meta, table="provider_info"):
    return sorted(meta.read("bronze_files", {"table_name": table}), key=lambda r: r["id"])


def test_first_load_writes_an_all_text_partition_with_lineage(
    store, meta, sink, put_landing, read_partition
):
    inv = put_landing(csv_text(ROWS, blank_tail=True))
    result = load(store, meta, sink, inv)

    assert result.status == "loaded"
    frame = read_partition("provider_info", "2025-06")
    assert frame.height == 2
    assert frame["CMS Certification Number (CCN)"].to_list() == ["015009", "15E123"]
    assert frame.schema["Provider Name"] == pl.String
    assert frame["_vintage_month"].to_list() == ["2025-06"] * 2
    assert frame["_row_number"].to_list() == [1, 2]
    assert store.exists("bronze/provider_info/vintage=2025-06/part-0.parquet")


def test_the_bronze_files_row_records_the_file(store, meta, sink, put_landing):
    inv = put_landing(csv_text(ROWS, blank_tail=True))
    load(store, meta, sink, inv)

    (row,) = files(meta)
    assert (row["status"], row["row_count"], row["blank_rows"], row["encoding"]) == (
        "loaded",
        2,
        1,
        "utf-8",
    )
    assert row["zip_sha256"] == "a" * 64 and row["run_id"] == "R1"
    assert row["header_hash"] == inv["header_hash"]
    assert row["content_hash"] and row["content_unchanged"] is None and row["object_uri"]


def test_a_cp1252_file_is_decoded_not_mangled(store, meta, sink, put_landing, read_partition):
    raw = csv_text([("015009", "St. Mary\x92s Home")]).encode("latin-1")
    load(store, meta, sink, put_landing(raw))

    assert read_partition("provider_info", "2025-06")["Provider Name"].to_list() == [
        "St. Mary’s Home"
    ]
    assert files(meta)[0]["encoding"] == "cp1252"


def test_the_same_zip_and_header_are_skipped_without_a_new_row(store, meta, sink, put_landing):
    inv = put_landing(csv_text(ROWS))
    load(store, meta, sink, inv)
    again = load(store, meta, sink, inv, run_id="R2")

    assert again.status == "skipped"
    assert len(files(meta)) == 1


def test_a_republished_zip_reloads_the_partition(store, meta, sink, put_landing, read_partition):
    load(store, meta, sink, put_landing(csv_text(ROWS)))
    revised = put_landing(csv_text([ROWS[0], ("15E123", "Pine Court II")]), zip_sha="b" * 64)
    result = load(store, meta, sink, revised, run_id="R2")

    assert result.status == "reloaded"
    assert read_partition("provider_info", "2025-06")["Provider Name"].to_list() == [
        "Oak Manor",
        "Pine Court II",
    ]
    assert [r["status"] for r in files(meta)] == ["loaded", "reloaded"]


def test_a_partition_without_a_bronze_files_row_is_reloaded(store, meta, sink, put_landing):
    inv = put_landing(csv_text(ROWS))
    load(store, meta, sink, inv)
    meta.delete_where(
        "bronze_files", {"table_name": "provider_info"}
    )  # crash between write and record

    assert load(store, meta, sink, inv, run_id="R2").status == "reloaded"


def test_a_missing_partition_is_reloaded_even_if_the_row_says_loaded(
    store, meta, sink, put_landing, drop_object
):
    inv = put_landing(csv_text(ROWS))
    load(store, meta, sink, inv)
    drop_object("bronze/provider_info/vintage=2025-06/part-0.parquet")

    assert load(store, meta, sink, inv, run_id="R2").status == "reloaded"


def test_row_count_mismatch_fails_closed(store, meta, sink, put_landing):
    inv = put_landing(csv_text(ROWS), row_count=99)
    result = load(store, meta, sink, inv)

    assert result.status == "failed" and "row count" in result.detail
    assert not store.exists("bronze/provider_info/vintage=2025-06/part-0.parquet")
    (row,) = files(meta)
    assert row["status"] == "failed" and "row count" in row["detail"]


def test_a_header_that_differs_from_the_inventory_fails_closed(store, meta, sink, put_landing):
    result = load(store, meta, sink, put_landing(csv_text(ROWS), header="0" * 16))
    assert result.status == "failed" and "header" in result.detail


def test_undecodable_bytes_fail_closed(store, meta, sink, put_landing):
    bad = csv_text([("015009", "Bad\x81Name")]).encode("latin-1")
    result = load(store, meta, sink, put_landing(bad, row_count=1, header="0" * 16))
    assert result.status == "failed" and "UTF-8" in result.detail


def test_a_missing_landing_file_fails_closed(store, meta, sink, put_landing):
    inv = put_landing(csv_text(ROWS))
    inv["extracted_key"] = "landing/cms_nh/extracted/gone.csv"
    result = load(store, meta, sink, inv)
    assert result.status == "failed" and files(meta)[0]["status"] == "failed"


def test_a_failed_file_is_retried_next_run_and_then_loads(store, meta, sink, put_landing):
    load(store, meta, sink, put_landing(csv_text(ROWS), row_count=99))
    fixed = put_landing(csv_text(ROWS))
    result = load(store, meta, sink, fixed, run_id="R2")

    assert result.status == "loaded"
    assert [r["status"] for r in files(meta)] == ["failed", "loaded"]


def test_a_failed_reload_leaves_the_old_partition_untouched(
    store, meta, sink, put_landing, read_partition
):
    load(store, meta, sink, put_landing(csv_text(ROWS)))
    broken = put_landing(csv_text(ROWS[:1]), zip_sha="b" * 64, row_count=99)
    assert load(store, meta, sink, broken, run_id="R2").status == "failed"

    assert read_partition("provider_info", "2025-06").height == 2
    assert read_partition("provider_info", "2025-06")["_zip_sha256"][0] == "a" * 64


def test_content_unchanged_compares_with_the_previous_vintage_ignoring_processing_date(
    store, meta, sink, put_landing
):
    load(store, meta, sink, put_landing(csv_text(ROWS), vintage="2025-06"))
    same = put_landing(
        csv_text(ROWS, processing="2025-07-01"),
        vintage="2025-07",
        processing_date="2025-07-01",
        zip_sha="c" * 64,
    )
    load(store, meta, sink, same, run_id="R2")
    changed = put_landing(
        csv_text(ROWS[:1], processing="2025-08-01"),
        vintage="2025-08",
        processing_date="2025-08-01",
        zip_sha="d" * 64,
    )
    load(store, meta, sink, changed, run_id="R3")

    june, july, august = (r["content_unchanged"] for r in files(meta))
    assert (june, july, august) == (None, True, False)


def test_a_table_without_a_processing_date_gets_a_null_lineage_date(
    store, meta, sink, put_landing, read_partition
):
    inv = put_landing(csv_text(ROWS), table="citation_lookup", processing_date=None)
    load(store, meta, sink, inv)
    assert read_partition("citation_lookup", "2025-06")["_processing_date"].to_list() == [
        None,
        None,
    ]
