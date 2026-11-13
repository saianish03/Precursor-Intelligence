"""configs/ml.yaml, configs/reason_labels.yaml and MlConfig (docs/ML_PIPELINE_SPEC.md §0.5, Appendix B)."""

import pytest
import yaml

from precursorintelligence.common.config import (
    ConfigError,
    load_feature_registry,
    load_labels_config,
    load_ml_config,
    load_qm_measures_config,
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
    assert cfg.features.exclude[:2] == ["era", "penalty_regime_2025_10"]
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


def test_quality_features_that_act_as_a_month_key_or_change_scale_stay_out():
    # Measured on the ml-dev silver (ml.yaml): qm_def_post_* and qm_age_months_* are one value per t; the
    # bridged measures and 401 / 451 change scale with their 2025-02 / 2026-02 definitions.
    cfg, qm = load_ml_config(), load_qm_measures_config()
    bridged = {f"qm_{n}" for n, m in qm.measures.items() if len(m.codes) > 1}
    month_keys = {*qm.definition_changes, "qm_age_months_mds", "qm_age_months_claims"}
    assert month_keys | bridged | {"qm_ls_adl_decline", "qm_ls_mobility_decline"} <= set(
        cfg.features.exclude
    )


# ------------------------------------------------------------------ members.lgbm and tuning.space


def test_an_unknown_lgbm_parameter_is_a_config_error(tmp_path):
    data = _repo()
    data["members"]["lgbm"]["num_leave"] = 31  # LightGBM would ignore it silently
    with pytest.raises(ConfigError, match="num_leave"):
        load_ml_config(_write(tmp_path, data))


def test_lgbm_needs_n_estimators(tmp_path):
    data = _repo()
    del data["members"]["lgbm"]["n_estimators"]
    with pytest.raises(ConfigError, match="n_estimators"):
        load_ml_config(_write(tmp_path, data))


def test_the_repo_lgbm_parameters_are_typed():
    lgbm = load_ml_config().members.lgbm
    assert lgbm.n_estimators == 1265 and lgbm.seed == 42 and lgbm.deterministic is True
    assert set(lgbm.as_dict()) == set(_repo()["members"]["lgbm"])


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("num_leave", {"kind": "int", "low": 8, "high": 128}, "num_leave"),
        ("n_estimators", {"kind": "int", "low": 100, "high": 2000}, "n_estimators"),
        (
            "lr_C",
            {"kind": "categorical", "choices": [0.1, 1.0], "low": 0.1, "high": 1.0},
            "only choices",
        ),
        ("learning_rate", {"kind": "float", "low": 0.005}, "low and high"),
        ("num_leaves", {"kind": "int", "low": 128, "high": 8}, "tuning range"),
        ("subsample", {"kind": "float", "low": 0.5, "high": 0.5}, "tuning range"),
        ("reg_alpha", {"kind": "float", "low": 0.0, "high": 10.0, "log": True}, "low > 0"),
    ],
)
def test_bad_tuning_space_entries_are_config_errors(tmp_path, key, value, message):
    data = _repo()
    data["tuning"]["space"][key] = value
    with pytest.raises(ConfigError, match=message):
        load_ml_config(_write(tmp_path, data))


def test_the_repo_tuning_space_is_the_spec_space():
    # spec §13 (PR #49's boost_models.py); log scale only on learning_rate, reg_alpha and reg_lambda
    space = load_ml_config().tuning.space
    assert {k: (v.kind, v.low, v.high, v.log) for k, v in space.items() if k != "lr_C"} == {
        "learning_rate": ("float", 0.005, 0.1, True),
        "num_leaves": ("int", 8, 128, False),
        "max_depth": ("int", 3, 12, False),
        "min_child_samples": ("int", 20, 1000, False),
        "min_split_gain": ("float", 0.0, 1.0, False),
        "subsample": ("float", 0.5, 1.0, False),
        "colsample_bytree": ("float", 0.3, 1.0, False),
        "reg_alpha": ("float", 0.001, 10.0, True),
        "reg_lambda": ("float", 0.001, 30.0, True),
        "cat_smooth": ("float", 1.0, 100.0, False),
    }
    assert space["lr_C"].kind == "categorical" and space["lr_C"].choices == [0.03, 0.1, 0.3, 1.0]
