"""Fixtures for the validation tests: a local object store and helpers that put small silver and gold tables."""

import pytest
from validation_fakes import parquet

from precursorintelligence.common.io.object_store import LocalObjectStore


@pytest.fixture
def store(tmp_path):
    return LocalObjectStore(tmp_path / "s")


@pytest.fixture
def put_silver(store):
    def put(table, vintage, df):
        store.put_bytes(
            f"silver/{table}/vintage={vintage}/part-0.parquet", parquet(df), overwrite=True
        )

    return put


@pytest.fixture
def put_gold(store):
    def put(table, df):
        store.put_bytes(f"gold/{table}/part-0.parquet", parquet(df), overwrite=True)

    return put
