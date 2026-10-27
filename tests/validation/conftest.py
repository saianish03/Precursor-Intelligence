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


@pytest.fixture
def env_file(tmp_path):
    """An env yaml over the same store as the `store` fixture and a SQLite metadata DB."""
    import yaml

    path = tmp_path / "env.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "env": "test-validation",
                "object_store": {"kind": "local", "root": str(tmp_path / "s")},
                "metadata_store": {"url": f"sqlite:///{tmp_path / 'm.db'}"},
                "logging": {"log_dir": str(tmp_path / "logs")},
            }
        )
    )
    return path


@pytest.fixture
def ctx(env_file):
    from precursorintelligence.common.config import load_settings
    from precursorintelligence.ingestion.context import build_context

    return build_context(load_settings("dev", env_path=env_file), run_id="R1")


@pytest.fixture
def cfg(tmp_path):
    from precursorintelligence.common.config import ValidationConfig

    return ValidationConfig.model_validate(
        {
            "schema_dir": str(tmp_path / "contracts"),
            "low_cardinality_max": 25,
            "top_values": 5,
            "layers": {
                "silver": {"partition_key": "vintage", "tables": ["citations_by_vintage"]},
                "gold": {"month_column": "t", "tables": ["spine"]},
            },
            "drift": {
                "row_count_change_pct": 10,
                "null_rate_jump_pts": 5,
                "mean_shift_std": 0.5,
                "skip_columns": ["y_*", "status_*", "split_*"],
            },
        }
    )
