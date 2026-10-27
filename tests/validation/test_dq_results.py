"""The dq_results metadata table the validate command writes (docs/DATA_PIPELINE_SPEC.md §15.2)."""

import pytest

from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore


@pytest.fixture
def meta(tmp_path):
    store = SqlAlchemyMetadataStore(f"sqlite:///{tmp_path / 'meta.db'}")
    store.ensure_schema()
    return store


def result_row(**over):
    row = {
        "run_id": "R1",
        "node": "validation",
        "layer": "silver",
        "table_name": "citations_by_vintage",
        "vintage_month": "2026-08",
        "check_name": "allowed_values",
        "column_name": "scope_severity",
        "severity": "error",
        "observed": '["Z"]',
        "expected": '["A", "B"]',
    }
    return row | over


def test_dq_results_is_append_only_history(meta):
    meta.append("dq_results", [result_row(), result_row(run_id="R2", severity="warning")])
    rows = sorted(
        meta.read("dq_results", {"table_name": "citations_by_vintage"}), key=lambda r: r["id"]
    )
    assert [(r["run_id"], r["severity"]) for r in rows] == [("R1", "error"), ("R2", "warning")]


def test_a_table_level_result_has_no_column(meta):
    meta.append("dq_results", [result_row(check_name="row_count_change", column_name=None)])
    (row,) = meta.read("dq_results")
    assert row["column_name"] is None and row["check_name"] == "row_count_change"
