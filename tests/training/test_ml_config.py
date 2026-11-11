"""configs/ml.yaml, configs/reason_labels.yaml and MlConfig (docs/ML_PIPELINE_SPEC.md §0.5, Appendix B)."""

import pytest
import yaml

from precursorintelligence.common.config import (
    ConfigError,
    load_feature_registry,
    load_labels_config,
    load_ml_config,
    project_root,
)


def _write(tmp_path, data):
    path = tmp_path / "ml.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _repo():
    return yaml.safe_load((project_root() / "configs" / "ml.yaml").read_text())


def test_the_repo_config_has_the_spec_defaults():
    cfg = load_ml_config()
    assert cfg.label == "g12" and cfg.label in load_labels_config().labels
    assert (cfg.windows.train_months, cfg.windows.min_train_months, cfg.windows.eval_gap) == (
        11,
        6,
        "production_lag",
    )
    assert cfg.folds.k == 5 and cfg.tuning.folds == 3
    assert cfg.features.exclude == ["era", "penalty_regime_2025_10"]
    assert cfg.calibration.method == "platt" and cfg.ensemble.weights == {"lgbm": 0.5, "lr": 0.5}
    assert cfg.outputs.root_prefix == "" and cfg.outputs.run_prefix == "ml/runs"
    assert cfg.allow_unverified is True  # until the data spec's §11 gate exists (ML-9)


def test_the_repo_config_names_files_and_features_that_exist():
    cfg = load_ml_config()
    model = {f.name for f in load_feature_registry().features if f.model_use == "model"}
    every = {f.name for f in load_feature_registry().features}
    assert set(cfg.features.exclude) <= model
    assert set(cfg.features.regime_sensitive) <= every
    assert (project_root() / cfg.bias.slices_config).is_file()
    assert (project_root() / cfg.explain.labels).is_file()


def test_reason_labels_has_one_entry_per_model_feature_in_registry_order():
    labels = yaml.safe_load((project_root() / "configs" / "reason_labels.yaml").read_text())
    model = [f.name for f in load_feature_registry().features if f.model_use == "model"]
    assert list(labels["labels"]) == model


def test_an_unknown_key_is_a_config_error(tmp_path):
    data = _repo()
    data["windows"]["train_month"] = 11  # a typo must not be ignored
    with pytest.raises(ConfigError, match="train_month"):
        load_ml_config(_write(tmp_path, data))


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("windows", "min_train_months", 12, "min_train_months is larger"),
        ("windows", "eval_gap", "none", "eval_gap"),
        ("folds", "k", 1, "k"),
        ("outputs", "root_prefix", "ml-dev", "root_prefix"),
        ("outputs", "root_prefix", "/ml-dev/", "root_prefix"),
        ("outputs", "run_prefix", "/ml/runs", "relative object key"),
        ("outputs", "status_key", "status/../x.json", "relative object key"),
        ("calibration", "method", "isotonic", "method"),
        ("selection", "candidates", ["lgbm", "lgbm"], "repeats"),
        ("features", "exclude", ["era", "era"], "repeats"),
    ],
)
def test_bad_values_are_config_errors(tmp_path, section, key, value, message):
    data = _repo()
    data[section][key] = value
    with pytest.raises(ConfigError, match=message):
        load_ml_config(_write(tmp_path, data))


def test_a_nested_prefix_is_accepted(tmp_path):
    data = _repo()
    data["outputs"]["root_prefix"] = "ml-dev/trial/"
    assert load_ml_config(_write(tmp_path, data)).outputs.root_prefix == "ml-dev/trial/"
