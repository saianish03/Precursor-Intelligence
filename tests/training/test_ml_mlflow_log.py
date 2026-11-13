"""MLflow tracking helpers (docs/ML_PIPELINE_SPEC.md §14): recipe hash, tags, secret refusal, the pyfunc package."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from precursorintelligence.common.config import FeatureSpec, LrParams, load_ml_config
from precursorintelligence.training import mlflow_log
from precursorintelligence.training.contracts import InputManifest, PlattCalibration
from precursorintelligence.training.ensemble import PrecursorModel, build_meta
from precursorintelligence.training.members import LgbmMember, LrMember, categorical_levels


def _spec(name: str, dtype: str = "f64") -> FeatureSpec:
    return FeatureSpec(
        name=name,
        family="x",
        source_table="x",
        formula="x",
        first_available_vintage="2022-01",
        dtype=dtype,  # type: ignore[arg-type]
        model_use="model",
    )


FEATURES = [_spec("a"), _spec("b"), _spec("c"), _spec("state", "str")]


def _manifest() -> InputManifest:
    return InputManifest(
        ml_run_id="manual__2026-11-12",
        gold_run_id="20261107T014704Z-c299c0",
        v="2026-10",
        v_labels="2024-02",
        dataset_version="dv-abc123",
        verified=True,
        merged_sha256="f" * 64,
        gold_git_commit=None,
        gold_config_sha256={},
        flags={},
        label="y_g12",
        features=[f.name for f in FEATURES],
        excluded_features=[],
        features_sha256="0" * 64,
        folds=3,
        rows={"train": 100},
        files={},
        frames_sha256={},
        windows_sha256="1" * 64,
        leakage_flags=[],
        regime_flags=[],
        pinned_at="2026-11-12T00:00:00Z",
    )


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    uri = f"sqlite:///{(tmp_path / 'mlruns.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.chdir(tmp_path)  # a default ./mlruns artifact root lands in tmp_path
    mlflow.set_tracking_uri(uri)
    yield uri
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- recipe hash


def test_recipe_hash_is_stable_and_tracks_features() -> None:
    h = mlflow_log.recipe_hash(FEATURES)
    assert len(h) == 64
    assert h == mlflow_log.recipe_hash(FEATURES)
    assert mlflow_log.recipe_hash(FEATURES[:-1]) != h
    retyped = [*FEATURES[:-1], _spec("state", "i64")]
    assert mlflow_log.recipe_hash(retyped) != h
    assert mlflow_log.recipe_hash(list(reversed(FEATURES))) != h


def test_recipe_hash_tracks_configs(tmp_path: Path) -> None:
    ml = tmp_path / "ml.yaml"
    ml.write_text("a: 1\n")
    h = mlflow_log.recipe_hash(FEATURES, ml_config_path=ml)
    ml.write_text("a: 2\n")
    assert mlflow_log.recipe_hash(FEATURES, ml_config_path=ml) != h
    missing = tmp_path / "none.yaml"
    reasons = tmp_path / "reasons.yaml"
    reasons.write_text("r: 1\n")
    without = mlflow_log.recipe_hash(FEATURES, ml_config_path=ml, reason_labels_path=missing)
    assert (
        mlflow_log.recipe_hash(FEATURES, ml_config_path=ml, reason_labels_path=reasons) != without
    )


# --------------------------------------------------------------------------- tags and secrets


def test_base_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_GIT_SHA", "deadbeef")
    monkeypatch.delenv("PI_PIPELINE_IMAGE", raising=False)
    monkeypatch.setenv("PI_DB_PASSWORD", "hunter2")
    tags = mlflow_log.base_tags(_manifest(), ml_run_id="r1", recipe_hash="h1")
    assert set(tags) == {
        "ml_run_id", "gold_run_id", "v", "v_labels", "dataset_version", "gold_verified",
        "gold_merged_sha256", "git_sha", "image", "recipe_hash", "decision", "mode",
    }  # fmt: skip
    assert tags["git_sha"] == "deadbeef" and tags["image"] == ""
    assert tags["gold_verified"] == "true" and tags["decision"] == "RETRAIN"
    assert all(isinstance(v, str) for v in tags.values())
    assert "hunter2" not in tags.values()


@pytest.mark.parametrize(
    "key", ["db_password", "SECRET", "gcs.Token", "my_credentials", "OPENAI_API_KEY"]
)
def test_log_params_refuses_secrets(key: str) -> None:
    with pytest.raises(ValueError, match="secret"):
        mlflow_log.log_params({key: "x"})
    with pytest.raises(ValueError, match="secret"):
        mlflow_log.log_params({"outer": {key: "x"}})


def test_disabled_tracking_is_a_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)
    assert not mlflow_log.tracking_enabled()
    cfg = load_ml_config()
    with mlflow_log.task_run(cfg, "train", {"a": "b"}) as run_id:
        assert run_id is None
        mlflow_log.log_params({"x": 1})
        mlflow_log.log_metrics({"m": 1.0})
        mlflow_log.log_gold_input(_manifest(), "gs://bucket/gold.parquet")
        mlflow_log.log_json_artifact({"a": 1}, "x.json")
        assert mlflow_log.log_package(PrecursorModel(), "pkg", pd.DataFrame()) == ""
    assert mlflow.active_run() is None
    assert not (tmp_path / "mlruns").exists()


# --------------------------------------------------------------------------- a tracked run


def test_task_run_logs_everything(tracking: str) -> None:
    cfg = load_ml_config()
    manifest = _manifest()
    tags = mlflow_log.base_tags(manifest, ml_run_id="r1", recipe_hash="h1")
    with mlflow_log.task_run(cfg, "train", tags) as run_id:
        assert run_id is not None
        mlflow_log.log_params({"lgbm": {"num_leaves": 31, "lr": 0.05}, "long": "x" * 7000})
        mlflow_log.log_metrics({"pr_auc": 0.48, "nan": float("nan"), "none": None})  # type: ignore[dict-item]
        mlflow_log.log_gold_input(manifest, "gs://bucket/gold/merged.parquet")
        mlflow_log.log_json_artifact(manifest, "manifest.json")
        mlflow_log.log_json_artifact({"k": [1, 2]}, "selection.json")
    client = MlflowClient(tracking)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "train"
    assert client.get_experiment(run.info.experiment_id).name == cfg.registry.experiment
    assert run.data.tags["task"] == "train"
    assert run.data.tags["ml_run_id"] == "r1" and run.data.tags["recipe_hash"] == "h1"
    assert run.data.params["lgbm.num_leaves"] == "31"
    assert len(run.data.params["long"]) == 6000
    assert run.data.metrics == {"pr_auc": 0.48}
    inputs = run.inputs.dataset_inputs
    assert len(inputs) == 1
    ds = inputs[0].dataset
    assert ds.name == "gold" and ds.digest == "dv-abc123"
    assert "gs://bucket/gold/merged.parquet" in ds.source
    assert {t.key: t.value for t in inputs[0].tags}.get("mlflow.data.context") == "training"
    names = {a.path for a in client.list_artifacts(run_id)}
    assert {"manifest.json", "selection.json"} <= names
    local = client.download_artifacts(run_id, "manifest.json")
    assert json.loads(Path(local).read_text())["dataset_version"] == "dv-abc123"


def test_failed_task_run_is_marked_failed(tracking: str) -> None:
    cfg = load_ml_config()
    with pytest.raises(RuntimeError), mlflow_log.task_run(cfg, "evaluate", {}) as run_id:
        raise RuntimeError("boom")
    assert MlflowClient(tracking).get_run(run_id).info.status == "FAILED"
    assert mlflow.active_run() is None


# --------------------------------------------------------------------------- the pyfunc package


def _tiny_model() -> tuple[PrecursorModel, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 120
    x = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n)),
            "state": rng.choice(["MA", "NY", "CA"], size=n).astype(object),
        }
    )
    y = (x["a"] + 0.5 * (x["state"] == "NY") + rng.normal(scale=0.5, size=n) > 0).astype(float)
    lr = LrMember.fit(
        x, y.to_numpy(), FEATURES, LrParams(C=0.3, max_iter=200, onehot_min_frequency=1)
    )
    lgbm = LgbmMember.fit(
        x,
        y.to_numpy(),
        FEATURES,
        {
            "n_estimators": 20,
            "num_leaves": 4,
            "seed": 1,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 1,
            "min_child_samples": 5,
        },
    )
    meta = build_meta(
        "ensemble",
        FEATURES,
        categorical_levels(x, FEATURES),
        {"lgbm": 0.6, "lr": 0.4},
        {"eval": ["2023-01"], "refit": ["2023-01"]},
        "h1",
    )
    platt = PlattCalibration(a=1.0, b=0.0, identity=False, fit_rows=10, fit_months=["2023-01"])
    return PrecursorModel(meta, platt, lgbm=lgbm, lr=lr), x


def test_log_package_round_trip(tracking: str) -> None:
    model, x = _tiny_model()
    cfg = load_ml_config()
    with mlflow_log.task_run(cfg, "train", {}):
        uri = mlflow_log.log_package(model, "eval_fit_ensemble", x)
    assert uri.startswith("models:/") or uri.startswith("runs:/")
    loaded = mlflow.pyfunc.load_model(uri)
    assert isinstance(loaded.unwrap_python_model(), PrecursorModel)
    expected = model.predict(None, x)
    got = loaded.predict(x)
    assert list(got.columns) == ["p", "logit", "logit_raw", "logit_lgbm", "logit_lr"]
    assert np.allclose(got.to_numpy(dtype=float), expected.to_numpy(dtype=float))
    assert loaded.metadata.get_input_schema().input_names() == ["a", "b", "c", "state"]


def test_log_package_without_run_returns_empty(tracking: str) -> None:
    model, x = _tiny_model()
    assert mlflow_log.log_package(model, "pkg", x) == ""
