"""evaluation/registry.py on a throwaway MLflow store (docs/ML_PIPELINE_SPEC.md §9.2)."""

from precursorintelligence.evaluation.registry import MlflowRegistry, default_registry
from precursorintelligence.training import mlflow_log
from precursorintelligence.training.contracts import RunPaths
from precursorintelligence.training.data import read_frame, to_model_frame


def test_no_tracking_means_no_registry(trained):
    assert default_registry(trained.cfg) is None


def test_register_find_alias_tag_and_load(trained, tracking):
    model = trained.train.refit_model
    sample = to_model_frame(
        read_frame(trained.store(), RunPaths.of(trained.cfg, trained.run_id), "score").head(20),
        model.features,
    )
    with mlflow_log.task_run(trained.cfg, "train", {}):
        uri = mlflow_log.log_package(model, "refit_x", sample)

    reg = default_registry(trained.cfg)
    assert isinstance(reg, MlflowRegistry)
    assert reg.by_alias("champion") is None and reg.find_by_ml_run_id("r1") is None

    v1 = reg.register(uri, {"ml_run_id": "r1", "candidate": "lgbm"})
    v2 = reg.register(uri, {"ml_run_id": "r2"})
    assert (v1.version, v2.version) == (1, 2)
    assert reg.find_by_ml_run_id("r1").version == 1

    reg.set_tags(1, {"gate_outcome": "pass"})
    reg.set_alias("champion", 1)
    champ = reg.by_alias("champion")
    assert champ.version == 1 and champ.tags["gate_outcome"] == "pass"
    assert reg.get(99) is None
    reg.delete_alias("champion")
    reg.delete_alias("champion")  # not set: no error
    assert reg.by_alias("champion") is None
    assert not hasattr(reg, "load_model")  # MLflow's copy of a package is never unpickled (§9.4)


class _Client:
    """An MlflowClient stand-in that raises the given MLflow error on every call."""

    def __init__(self, exc):
        self.exc = exc

    def __getattr__(self, name):
        def fail(*a, **kw):
            raise self.exc

        return fail


def test_only_a_missing_resource_reads_as_none():
    import pytest
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import (
        INTERNAL_ERROR,
        INVALID_PARAMETER_VALUE,
        RESOURCE_DOES_NOT_EXIST,
    )

    reg = MlflowRegistry.__new__(MlflowRegistry)
    reg.name = "m"
    missing = MlflowException(
        "Model Version (name=m, version=1) not found", RESOURCE_DOES_NOT_EXIST
    )
    reg.client = _Client(missing)
    assert reg.by_alias("champion") is None and reg.get(1) is None
    reg.client = _Client(
        MlflowException("Registered model alias champion not found.", INVALID_PARAMETER_VALUE)
    )
    assert reg.by_alias("champion") is None
    # a server error is not "no champion": it propagates, so the task fails and is retried
    reg.client = _Client(MlflowException("bad gateway", INTERNAL_ERROR))
    for call in (
        lambda: reg.by_alias("champion"),
        lambda: reg.get(1),
        lambda: reg.find_by_ml_run_id("r"),
    ):
        with pytest.raises(MlflowException):
            call()


class _RecordingClient:
    """An MlflowClient stand-in that records create_model_version's arguments."""

    def __init__(self):
        self.calls = []

    def get_registered_model(self, name):
        return object()

    def create_model_version(self, name, source, **kw):
        from types import SimpleNamespace

        self.calls.append((source, kw.get("model_id")))
        return SimpleNamespace(version="1", tags={})


def test_a_logged_model_is_registered_with_its_model_id():
    # MLflow 3's log_model returns models:/<model id>; a tracking server refuses that source without the model id
    # ("Invalid model version source"), while the SQLite store of the other tests accepts it.
    reg = MlflowRegistry.__new__(MlflowRegistry)
    reg.name, reg.client = "m", _RecordingClient()
    reg.register("models:/m-6fcbd66caa67410bbc8b7ace4b5a93f1", {"ml_run_id": "r1"})
    reg.register("runs:/abc/refit_x", {"ml_run_id": "r2"})
    assert reg.client.calls == [
        ("models:/m-6fcbd66caa67410bbc8b7ace4b5a93f1", "m-6fcbd66caa67410bbc8b7ace4b5a93f1"),
        ("runs:/abc/refit_x", None),
    ]
