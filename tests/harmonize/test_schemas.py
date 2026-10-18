"""The two metadata tables bronze writes (docs/DATA_PIPELINE_SPEC.md §3.1 step 9 and §15.2)."""

import pytest

from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore


@pytest.fixture
def meta(tmp_path):
    store = SqlAlchemyMetadataStore(f"sqlite:///{tmp_path / 'meta.db'}")
    store.ensure_schema()
    return store


def loaded_row(**over):
    row = {
        "run_id": "R1",
        "table_name": "provider_info",
        "vintage_month": "2025-06",
        "entry_id": "current:2025-06-26:nh.zip",
        "member_path": "NH_ProviderInfo_Jun2025.csv",
        "zip_sha256": "a" * 64,
        "header_hash": "h" * 16,
        "status": "loaded",
        "row_count": 14700,
        "blank_rows": 1,
        "encoding": "cp1252",
        "content_hash": "c" * 64,
        "content_unchanged": False,
    }
    return row | over


def test_bronze_files_is_append_only_history(meta):
    meta.append("bronze_files", [loaded_row(status="failed", detail="row count mismatch")])
    meta.append("bronze_files", [loaded_row(run_id="R2")])
    rows = meta.read("bronze_files", {"table_name": "provider_info", "vintage_month": "2025-06"})
    assert [r["status"] for r in sorted(rows, key=lambda r: r["id"])] == ["failed", "loaded"]


def test_bronze_files_round_trips_flags_and_a_missing_content_unchanged(meta):
    meta.append("bronze_files", [loaded_row(), loaded_row(run_id="R2", content_unchanged=None)])
    first, second = sorted(meta.read("bronze_files"), key=lambda r: r["id"])
    assert first["content_unchanged"] is False and first["blank_rows"] == 1
    assert (
        second["content_unchanged"] is None
    )  # the first vintage of a table has nothing to compare to


def test_vintage_flags_hold_one_current_value_per_flag(meta):
    release = {"vintage_month": "2025-09", "flag": "stale_refresh", "table_name": ""}
    meta.upsert("vintage_flags", [release | {"value": False, "run_id": "R1"}])
    meta.upsert("vintage_flags", [release | {"value": True, "run_id": "R2"}])
    rows = meta.read("vintage_flags", {"vintage_month": "2025-09"})
    assert len(rows) == 1 and rows[0]["value"] is True and rows[0]["run_id"] == "R2"


def test_revised_flag_is_kept_per_table(meta):
    base = {"vintage_month": "2024-03", "flag": "revised_after_publication", "run_id": "R1"}
    meta.upsert(
        "vintage_flags",
        [
            base | {"table_name": "provider_info", "value": True},
            base | {"table_name": "penalties", "value": False},
        ],
    )
    assert {r["table_name"]: r["value"] for r in meta.read("vintage_flags")} == {
        "provider_info": True,
        "penalties": False,
    }
