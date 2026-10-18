"""Lineage columns and the fail-closed reconciliation of docs/DATA_PIPELINE_SPEC.md §3.1 steps 6 and 7."""

from datetime import UTC, date, datetime

import polars as pl
import pytest

from precursorintelligence.harmonize.lineage import ReconcileError, add_lineage, reconcile
from precursorintelligence.harmonize.reader import parse_csv_text
from precursorintelligence.ingestion.layout import header_hash, normalize_header

HEADER = ["CMS Certification Number (CCN)", "Provider Name"]
TEXT = (
    '"CMS Certification Number (CCN)","Provider Name"\r\n"015009","Oak"\r\n"15E123","Pine"\r\n\r\n'
)
INGESTED = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)


def inv_row(**over):
    row = {
        "table_name": "provider_info",
        "vintage_month": "2026-08",
        "processing_date": "2026-08-01",
        "zip_name": "nh_archive_2026-08.zip",
        "zip_sha256": "a" * 64,
        "member_path": "NH_ProviderInfo_Aug2026.csv",
        "header_hash": header_hash(normalize_header(HEADER)),
        "row_count": 3,  # two data rows plus the trailing blank line
    }
    return row | over


def test_lineage_columns_follow_the_data_columns_with_the_specified_types():
    out = add_lineage(parse_csv_text(TEXT).frame, inv_row(), INGESTED)
    lineage = out.columns[len(HEADER) :]
    assert lineage == [
        "_vintage_month",
        "_processing_date",
        "_source_zip",
        "_zip_sha256",
        "_source_file",
        "_header_hash",
        "_row_number",
        "_ingested_at",
    ]
    assert out.schema["_processing_date"] == pl.Date
    assert out.schema["_row_number"] == pl.Int64
    assert out.schema["_ingested_at"] == pl.Datetime("us", "UTC")
    assert out.schema["_vintage_month"] == pl.String


def test_lineage_values_come_from_the_inventory_row():
    row = add_lineage(parse_csv_text(TEXT).frame, inv_row(), INGESTED).row(0, named=True)
    assert row["_vintage_month"] == "2026-08"
    assert row["_processing_date"] == date(2026, 8, 1)
    assert row["_source_zip"] == "nh_archive_2026-08.zip"
    assert row["_zip_sha256"] == "a" * 64
    assert row["_source_file"] == "NH_ProviderInfo_Aug2026.csv"
    assert row["_ingested_at"] == INGESTED


def test_row_numbers_are_one_based_after_the_blank_rows_are_dropped():
    out = add_lineage(parse_csv_text(TEXT).frame, inv_row(), INGESTED)
    assert out["_row_number"].to_list() == [1, 2]


def test_processing_date_is_null_when_the_table_has_none():
    out = add_lineage(parse_csv_text(TEXT).frame, inv_row(processing_date=None), INGESTED)
    assert out["_processing_date"].to_list() == [None, None]
    assert out.schema["_processing_date"] == pl.Date


def test_data_columns_are_untouched_by_lineage():
    out = add_lineage(parse_csv_text(TEXT).frame, inv_row(), INGESTED)
    assert out["CMS Certification Number (CCN)"].to_list() == ["015009", "15E123"]


def test_reconcile_accepts_rows_plus_blank_rows_equal_to_the_inventory_count():
    reconcile(parse_csv_text(TEXT), inv_row())


def test_reconcile_rejects_a_row_count_mismatch():
    with pytest.raises(ReconcileError, match="row count"):
        reconcile(parse_csv_text(TEXT), inv_row(row_count=4))


def test_reconcile_rejects_a_header_that_differs_from_the_inventory():
    with pytest.raises(ReconcileError, match="header"):
        reconcile(parse_csv_text(TEXT), inv_row(header_hash="0" * 16))
