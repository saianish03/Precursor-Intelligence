"""Fixtures for the ML tests: the synthetic published gold of tests/ml/make_ml_gold.py and its stores."""

import sys
from pathlib import Path

import mlflow
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "ml"))  # the synthetic gold lives in tests/ml
from make_ml_gold import MlGold, make_ml_gold, ml_config  # noqa: E402

from precursorintelligence.common.config import (  # noqa: E402
    load_feature_registry,
    load_labels_config,
    load_splits_config,
)
from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore  # noqa: E402
from precursorintelligence.common.io.object_store import LocalObjectStore  # noqa: E402
from precursorintelligence.training.data import run_prepare  # noqa: E402


@pytest.fixture
def gold(tmp_path: Path) -> MlGold:
    return make_ml_gold(tmp_path / "store")


@pytest.fixture
def store(gold: MlGold) -> LocalObjectStore:
    return LocalObjectStore(gold.root, immutable=None)


@pytest.fixture
def meta(gold: MlGold) -> SqlAlchemyMetadataStore:
    m = SqlAlchemyMetadataStore(f"sqlite:///{(gold.root / 'ops' / 'metadata.db').as_posix()}")
    m.ensure_schema()
    return m


@pytest.fixture
def prepare():
    """run_prepare with the repo's configs and the fixture-sized ml config."""

    def _prepare(store, meta, run_id="manual__t", cfg=None, **kwargs):
        return run_prepare(
            store,
            meta,
            ml_run_id=run_id,
            cfg=cfg or ml_config(),
            labels_cfg=load_labels_config(),
            splits_cfg=load_splits_config(),
            registry=load_feature_registry(),
            **kwargs,
        )

    return _prepare


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A throwaway SQLite MLflow store; tracking is on while the test runs."""
    uri = f"sqlite:///{(tmp_path / 'mlruns.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.chdir(tmp_path)  # a default ./mlruns artifact root lands in tmp_path
    mlflow.set_tracking_uri(uri)
    yield uri
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(None)  # type: ignore[arg-type]
