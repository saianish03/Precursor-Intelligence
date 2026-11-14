"""evaluation/registry.py on a throwaway MLflow store (docs/ML_PIPELINE_SPEC.md §9.2)."""

import numpy as np

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

    loaded = reg.load_model(uri)
    assert np.allclose(loaded.predict_frame(sample)["p"], model.predict_frame(sample)["p"])
