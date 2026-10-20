"""Fixtures for the gold tests (the fake silver world lives in silver_fakes.py)."""

import polars as pl
import pytest
from silver_fakes import build_silver, write_silver

from precursorintelligence.common.io.object_store import LocalObjectStore


@pytest.fixture(scope="session")
def silver() -> dict[str, pl.DataFrame]:
    return build_silver()


@pytest.fixture
def silver_store(tmp_path, silver) -> LocalObjectStore:
    """The fake silver tables written at their spec paths under a temp root."""
    store = LocalObjectStore(tmp_path, immutable=None)
    write_silver(store, silver)
    return store
