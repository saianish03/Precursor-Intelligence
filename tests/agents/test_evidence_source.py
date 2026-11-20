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


def test_month_outside_the_vintage_range_refuses(store):
    write_evidence(store, v="2027-01")
    with pytest.raises(EvidenceError, match="not YYYY-MM in 2021-2026"):
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


def test_a_changed_file_is_refused_on_read(store):
    write_evidence(store)
    store.put_bytes(
        f"{PREFIX}v={V}/build={BUILD}/citations.parquet",
        toy_tables()["citations"].head(1).write_ipc(None).getvalue(),
        overwrite=True,
    )
    source = ParquetEvidenceSource.open(store, PREFIX)
    with pytest.raises(EvidenceError, match="does not match its sha256"):
        source.rows("citations", "ccn", ["015009"])


def test_a_missing_column_is_refused(store):
    tables = toy_tables()
    tables["citations"] = tables["citations"].drop("scope_severity")
    write_evidence(store, tables)
    source = ParquetEvidenceSource.open(store, PREFIX)
    with pytest.raises(EvidenceError, match="scope_severity"):
        source.rows("citations", "ccn", ["015009"])


def test_rows_of_another_gold_run_are_refused(store):
    tables = toy_tables()
    tables["facility_scores"] = tables["facility_scores"].with_columns(
        gold_run_id=pl.lit("gold-other")
    )
    write_evidence(store, tables)
    source = ParquetEvidenceSource.open(store, PREFIX)
    with pytest.raises(EvidenceError, match="another gold run"):
        source.rows("facility_scores", "ccn", ["015009"])


def test_a_numeric_ccn_is_refused_not_read_as_unscored(store):
    tables = toy_tables()
    tables["facility_scores"] = (
        tables["facility_scores"]
        .filter(pl.col("ccn") != "01A193")
        .with_columns(pl.col("ccn").cast(pl.Int64))
    )  # 015009 would become 15009
    write_evidence(store, tables)
    source = ParquetEvidenceSource.open(store, PREFIX)
    with pytest.raises(EvidenceError, match="wrong type"):
        source.rows("facility_scores", "ccn", ["015009"])
