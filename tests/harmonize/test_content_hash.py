"""Content hash and content_unchanged of docs/DATA_PIPELINE_SPEC.md §3.1 step 8."""

import polars as pl
import pytest

from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore
from precursorintelligence.harmonize.content import (
    content_hash,
    content_unchanged,
    previous_content_hash,
)


def frame(rows, processing="2026-08-01"):
    return pl.DataFrame(
        {
            "CMS Certification Number (CCN)": [r[0] for r in rows],
            "Penalty Type": [r[1] for r in rows],
            "Processing Date": [processing] * len(rows),
        }
    )


ROWS = [("015009", "Fine"), ("15E123", "Payment denial"), ("015009", "Fine")]


def test_hash_is_a_sha256_hex_digest():
    digest = content_hash(frame(ROWS))
    assert len(digest) == 64 and int(digest, 16) >= 0


def test_hash_does_not_depend_on_row_order():
    assert content_hash(frame(ROWS)) == content_hash(frame(list(reversed(ROWS))))


def test_hash_ignores_the_processing_date_column():
    assert content_hash(frame(ROWS, "2026-08-01")) == content_hash(frame(ROWS, "2026-09-01"))


def test_hash_changes_when_any_value_changes():
    changed = [ROWS[0], ("15E123", "Fine"), ROWS[2]]
    assert content_hash(frame(ROWS)) != content_hash(frame(changed))


def test_hash_keeps_exact_duplicate_rows_distinct_from_a_single_copy():
    assert content_hash(frame(ROWS)) != content_hash(frame(ROWS[:2]))


def test_hash_does_not_depend_on_how_the_rows_are_chunked():
    assert content_hash(frame(ROWS), chunk_rows=1) == content_hash(frame(ROWS))


def test_a_null_and_an_empty_string_hash_differently():
    nulls = frame([("015009", None)])
    empties = frame([("015009", "")])
    assert content_hash(nulls) != content_hash(empties)


@pytest.fixture
def meta(tmp_path):
    store = SqlAlchemyMetadataStore(f"sqlite:///{tmp_path / 'meta.db'}")
    store.ensure_schema()
    return store


def file_row(table, vintage, content_hash, status="loaded"):
    return {
        "table_name": table,
        "vintage_month": vintage,
        "content_hash": content_hash,
        "status": status,
    }


def test_the_previous_hash_is_the_nearest_earlier_published_vintage_of_the_same_table(meta):
    meta.append(
        "bronze_files",
        [
            file_row("penalties", "2025-05", "h05"),
            file_row("penalties", "2025-07", "h07"),
            file_row("penalties", "2025-08", "failed-hash", status="failed"),
            file_row("penalties", "2025-10", "h10"),
            file_row("provider_info", "2025-08", "other-table"),
        ],
    )
    assert previous_content_hash(meta, "penalties", "2025-09") == "h07"


def test_a_reloaded_row_counts_as_published_and_the_latest_attempt_wins(meta):
    meta.append(
        "bronze_files",
        [
            file_row("penalties", "2025-07", "old"),
            file_row("penalties", "2025-07", "new", status="reloaded"),
        ],
    )
    assert previous_content_hash(meta, "penalties", "2025-08") == "new"


def test_the_first_vintage_of_a_table_has_no_previous_hash(meta):
    meta.append("bronze_files", [file_row("penalties", "2025-07", "h07")])
    assert previous_content_hash(meta, "penalties", "2025-07") is None


def test_content_unchanged_is_unknown_without_a_previous_hash():
    assert content_unchanged("h1", None) is None
    assert content_unchanged("h1", "h1") is True
    assert content_unchanged("h1", "h2") is False
