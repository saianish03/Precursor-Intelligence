"""The metadata store's read-only copy in BigQuery (docs/DATA_PIPELINE_SPEC.md §0.6, §15.3): no network, a fake
client that records each load job."""

import json
from types import SimpleNamespace

import pytest
from google.api_core.exceptions import BadRequest
from sqlalchemy import Column, Float, MetaData, Table

from precursorintelligence.common.config import BigQueryCfg
from precursorintelligence.common.io.metadata_mirror import (
    NOT_MIRRORED,
    MirrorError,
    bq_type,
    mirror_metadata,
)
from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore
from precursorintelligence.common.io.schemas import TABLES

CFG = BigQueryCfg(project="proj", dataset="scores", meta_dataset="meta", location="us-central1")
MIRRORED = [name for name in TABLES if name not in NOT_MIRRORED]


class FakeBigQuery:
    """Records each load (destination, job config, location, the uploaded rows) and then holds that many rows.
    ``fail`` maps a table to the error its load raises; ``held_extra`` to rows BigQuery holds beyond the load."""

    def __init__(self, fail=None, held_extra=None):
        self.loads, self.held = [], {}
        self.fail, self.held_extra = fail or {}, held_extra or {}

    def load_table_from_file(self, file_obj, destination, *, job_config, location):
        table = destination.rsplit(".", 1)[1]
        rows = [json.loads(line) for line in file_obj.read().decode("utf-8").splitlines()]
        self.loads.append((destination, job_config, location, rows))
        if table in self.fail:
            raise self.fail[table]
        self.held[destination] = len(rows) + self.held_extra.get(table, 0)
        return SimpleNamespace(result=lambda: None)

    def get_table(self, destination):
        return SimpleNamespace(num_rows=self.held[destination])

    def load(self, table):
        return next(load for load in self.loads if load[0] == f"proj.meta.{table}")


@pytest.fixture
def meta():
    store = SqlAlchemyMetadataStore("sqlite:///:memory:")
    store.ensure_schema()
    store.append(
        "run_log",
        [
            {"run_id": "r1", "task": "bronze", "status": "ok", "detail": "naïve – unicode"},
            {"run_id": "r1", "task": "silver", "status": "error", "detail": None},
        ],
    )
    store.upsert(
        "vintage_flags",
        [{"vintage_month": "2025-09", "flag": "stale_refresh", "table_name": "*", "value": True}],
    )
    return store


def test_every_table_but_run_lock_is_replaced_whole_with_its_columns(meta):
    client = FakeBigQuery()
    out = mirror_metadata(meta, CFG, client)
    assert NOT_MIRRORED == ("run_lock",) and list(out) == MIRRORED
    assert [load[0] for load in client.loads] == [f"proj.meta.{name}" for name in MIRRORED]
    for destination, job_config, location, rows in client.loads:
        name = destination.rsplit(".", 1)[1]
        assert location == "us-central1"
        assert job_config.source_format == "NEWLINE_DELIMITED_JSON"
        assert job_config.write_disposition == "WRITE_TRUNCATE"  # the whole table, each run
        assert job_config.create_disposition == "CREATE_IF_NEEDED"
        assert [(f.name, f.field_type, f.mode) for f in job_config.schema] == [
            (c.name, bq_type(c), "NULLABLE") for c in TABLES[name].columns
        ]
        assert rows == meta.read(name) and out[name] == len(rows)


def test_the_rows_keep_their_values_and_types(meta):
    client = FakeBigQuery()
    mirror_metadata(meta, CFG, client)
    log = client.load("run_log")[3]
    assert [(r["id"], r["task"], r["detail"]) for r in log] == [
        (1, "bronze", "naïve – unicode"),
        (2, "silver", None),
    ]
    assert client.load("vintage_flags")[3][0]["value"] is True


def test_an_empty_table_is_emptied_with_its_columns(meta):
    # the row count is read back from BigQuery, so an empty load that left old rows would fail
    client = FakeBigQuery()
    out = mirror_metadata(meta, CFG, client)
    _, job_config, _, rows = client.load("vintage_registry")
    assert rows == [] and out["vintage_registry"] == 0
    assert len(job_config.schema) == len(TABLES["vintage_registry"].columns)
    with pytest.raises(MirrorError) as exc:
        mirror_metadata(meta, CFG, FakeBigQuery(held_extra={"vintage_registry": 3}))
    assert exc.value.failures == {"vintage_registry": "BigQuery holds 3 rows, the store 0"}


def test_a_failing_table_never_stops_the_others(meta):
    # the first table failing still lets every other table be mirrored. Any BigQuery error is retryable (exit 1):
    # a 400 can be temporary (resourceInUse, tableUnavailable), and a full replace is idempotent
    first = MIRRORED[0]
    with pytest.raises(MirrorError) as exc:
        mirror_metadata(meta, CFG, FakeBigQuery(fail={first: BadRequest("bad schema")}))
    assert set(exc.value.failures) == {first} and "BadRequest" in exc.value.failures[first]
    assert list(exc.value.loaded) == MIRRORED[1:]
    assert exc.value.retryable


def test_other_rows_in_bigquery_are_refused_not_retried(meta):
    with pytest.raises(MirrorError) as exc:
        mirror_metadata(meta, CFG, FakeBigQuery(held_extra={"run_log": 1}))
    assert exc.value.failures == {"run_log": "BigQuery holds 3 rows, the store 2"}
    assert not exc.value.retryable and "run_log" not in exc.value.loaded


def test_every_column_in_the_store_has_a_bigquery_type():
    # a new column type must get a mapping here, not a guess
    for table in TABLES.values():
        for column in table.columns:
            assert bq_type(column) in {"STRING", "INT64", "BOOL"}, (table.name, column.name)
    unmapped = Table("t", MetaData(), Column("x", Float)).c.x
    with pytest.raises(ValueError, match="t.x: no BigQuery type"):
        bq_type(unmapped)
