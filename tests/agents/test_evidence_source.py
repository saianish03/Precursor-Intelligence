"""E1: the source serves only the month in ``_current.json``, and only files that match their manifest."""

import json

import polars as pl
import pytest
from evidence_fakes import BUILD, GOLD, PREFIX, V, toy_tables, write_evidence

from precursorintelligence.agents.tools.source import EvidenceError, ParquetEvidenceSource


def test_reads_the_month_in_the_pointer_not_a_newer_folder(store):
    write_evidence(store)
    # A newer month whose build has not moved the pointer yet (e.g. still being written).
    store.put_bytes(f"{PREFIX}v=2026-11/facility_scores.parquet", b"partial")
    source = ParquetEvidenceSource.open(store, PREFIX)
    assert source.manifest.v == V
    assert source.manifest.gold_run_id == GOLD
    assert source.rows("facility_scores", "ccn", ["015009"])["p"].to_list() == [0.42]


def test_keeps_the_pointer_it_read(store):
    # The answer cache's key (D1) comes from the same read as the evidence, never a second read of the pointer.
    write_evidence(store)
    pointer = json.loads(store.open_read(f"{PREFIX}_current.json").read())
    source = ParquetEvidenceSource.open(store, PREFIX)
    assert source.pointer.model_dump() == pointer


def test_reads_never_touch_the_store_after_open(store):
    # A stuck store read can then only fail the instance's start, never hold up a tool call.
    m = write_evidence(store)
    source = ParquetEvidenceSource.open(store, PREFIX)
    for key in m["files"].values():
        store.put_bytes(key, b"gone", overwrite=True)
    assert source.rows("facility_scores", "ccn", ["015009"])["p"].to_list() == [0.42]


def test_no_pointer_refuses(store):
    with pytest.raises(EvidenceError, match="_current.json is missing"):
        ParquetEvidenceSource.open(store, PREFIX)


def test_pointer_and_manifest_must_agree(store):
    write_evidence(store)
    key = f"{PREFIX}v={V}/build={BUILD}/evidence_manifest.json"
    manifest = json.loads(store.open_read(key).read())
    manifest["built_at"] = "2026-10-17T00:00:00Z"  # rewritten after the pointer moved
    store.put_bytes(key, json.dumps(manifest).encode(), overwrite=True)
    with pytest.raises(EvidenceError, match="does not match the sha256"):
        ParquetEvidenceSource.open(store, PREFIX)


def test_a_month_after_2026_is_served(store):
    # The archive starts in 2021, but the monthly run keeps going: 2027-01 must not stop the agents.
    write_evidence(store, v="2027-01")
    assert ParquetEvidenceSource.open(store, PREFIX).pointer.v == "2027-01"


def test_a_month_that_is_not_yyyy_mm_refuses(store):
    write_evidence(store, v="2026-13")
    with pytest.raises(EvidenceError, match="not YYYY-MM"):
        ParquetEvidenceSource.open(store, PREFIX)


def test_manifest_may_not_point_at_another_month(store):
    tables = toy_tables()
    m = write_evidence(store, tables)
    files = dict(m["files"], citations=f"{PREFIX}v=2026-09/citations.parquet")
    write_evidence(store, tables, manifest_overrides={"files": files})
    with pytest.raises(EvidenceError, match="outside"):
        ParquetEvidenceSource.open(store, PREFIX)


def test_other_files_of_the_month_are_allowed(store):
    # build-evidence also writes the ops evidence of Agent 3 into the month's folder (E2, I1).
    tables = toy_tables()
    m = write_evidence(store, tables)
    ops = f"{PREFIX}v={V}/build={BUILD}/ops_status.json"
    files = dict(m["files"], ops_status=ops)
    write_evidence(store, tables, manifest_overrides={"files": files})
    assert ParquetEvidenceSource.open(store, PREFIX).manifest.files["ops_status"] == ops

    files = dict(
        m["files"], ops_status=f"{PREFIX}v={V}/build=ba9876543210/ops_status.json"
    )  # another build
    write_evidence(store, tables, manifest_overrides={"files": files})
    with pytest.raises(EvidenceError, match="outside"):
        ParquetEvidenceSource.open(store, PREFIX)


def test_a_missing_table_refuses(store):
    tables = toy_tables()
    m = write_evidence(store, tables)
    files = {k: v for k, v in m["files"].items() if k != "penalties"}
    write_evidence(store, tables, manifest_overrides={"files": files})
    with pytest.raises(EvidenceError, match="lacks the tables"):
        ParquetEvidenceSource.open(store, PREFIX)


def test_a_changed_file_is_refused_on_open(store):
    write_evidence(store)
    store.put_bytes(
        f"{PREFIX}v={V}/build={BUILD}/citations.parquet",
        toy_tables()["citations"].head(1).write_ipc(None).getvalue(),
        overwrite=True,
    )
    with pytest.raises(EvidenceError, match="does not match its sha256"):
        ParquetEvidenceSource.open(store, PREFIX)


def test_a_missing_column_is_refused(store):
    tables = toy_tables()
    tables["citations"] = tables["citations"].drop("scope_severity")
    write_evidence(store, tables)
    with pytest.raises(EvidenceError, match="scope_severity"):
        ParquetEvidenceSource.open(store, PREFIX)


def test_rows_of_another_gold_run_are_refused(store):
    tables = toy_tables()
    tables["facility_scores"] = tables["facility_scores"].with_columns(
        gold_run_id=pl.lit("gold-other")
    )
    write_evidence(store, tables)
    with pytest.raises(EvidenceError, match="another gold run"):
        ParquetEvidenceSource.open(store, PREFIX)


def test_a_numeric_ccn_is_refused_not_read_as_unscored(store):
    tables = toy_tables()
    tables["facility_scores"] = (
        tables["facility_scores"]
        .filter(pl.col("ccn") != "01A193")
        .with_columns(pl.col("ccn").cast(pl.Int64))
    )  # 015009 would become 15009
    write_evidence(store, tables)
    with pytest.raises(EvidenceError, match="wrong type"):
        ParquetEvidenceSource.open(store, PREFIX)


def test_a_null_in_a_required_column_is_refused(store):
    # A null the tools' records do not allow would fail one facility at a time inside a tool; refuse the month.
    tables = toy_tables()
    tables["citations"] = tables["citations"].with_columns(
        scope_severity=pl.when(pl.col("ccn") == "015009")
        .then(None)
        .otherwise(pl.col("scope_severity"))
    )
    write_evidence(store, tables)
    with pytest.raises(EvidenceError, match=r"nulls in \['scope_severity'\]"):
        ParquetEvidenceSource.open(store, PREFIX)


def test_a_null_in_an_optional_column_is_served(store):
    tables = toy_tables()
    tables["citations"] = tables["citations"].with_columns(correction_date=pl.lit(None, pl.Date))
    write_evidence(store, tables)
    assert ParquetEvidenceSource.open(store, PREFIX).rows("citations", "ccn", ["015009"]).height
