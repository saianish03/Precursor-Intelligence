"""training/tune.py on the synthetic gold: the proposal, the files, determinism, never the test frame
(docs/ML_PIPELINE_SPEC.md §13, §8.3, §20)."""

import json

import pytest
import yaml
from make_ml_gold import ml_config

from precursorintelligence.common.config import load_feature_registry
from precursorintelligence.training import MlInputError
from precursorintelligence.training.contracts import RunPaths
from precursorintelligence.training.data import feature_list, read_frame
from precursorintelligence.training.tune import (
    LR_C,
    PLOT_NAMES,
    SEARCH_SPACE,
    run_tune,
    tune,
)


def small_config():
    cfg = ml_config()
    tuning = cfg.tuning.model_copy(update={"timeout_minutes": 2.0, "early_stopping_rounds": 10})
    return cfg.model_copy(update={"tuning": tuning})


def test_the_proposal_is_in_the_space_and_written(store, meta, prepare):
    cfg = small_config()
    prepare(store, meta, run_id="manual__tune", cfg=cfg)
    registry = load_feature_registry()
    res = run_tune(
        store, ml_run_id="manual__tune", cfg=cfg, registry=registry, n_trials=3, max_rounds=20
    )

    proposal = yaml.safe_load(res.proposal_yaml)
    lgbm = proposal["members"]["lgbm"]
    for name, p in SEARCH_SPACE.items():
        if name == LR_C:
            continue
        assert p.low <= lgbm[name] <= p.high, name
    for fixed in ("seed", "deterministic", "force_col_wise", "num_threads", "subsample_freq"):
        assert lgbm[fixed] == cfg.members.lgbm[fixed]
    assert isinstance(lgbm["n_estimators"], int) and 1 <= lgbm["n_estimators"] <= 20
    assert lgbm == res.best_params
    assert proposal["members"]["lr"]["C"] == res.best_lr_C in SEARCH_SPACE[LR_C].choices
    assert proposal["cv_pr_auc"] == pytest.approx(res.best_value) and 0 < res.best_value <= 1
    assert proposal["n_trials"] == 3 and proposal["folds"] == cfg.tuning.folds == 3
    assert proposal["source"]["ml_run_id"] == "manual__tune"
    assert res.proposal_yaml.startswith("# Tuning proposal") and "reviewed PR" in res.proposal_yaml

    assert len(res.trials) == 3
    for attr in ("best_iterations", "lgbm_pr_auc", "lr_pr_auc"):
        assert f"user_attrs_{attr}" in res.trials.columns

    paths = RunPaths.of(cfg, "manual__tune")
    assert store.open_read(paths.tuning_proposal).read().decode() == res.proposal_yaml
    assert store.exists(f"{paths.tuning_dir}trials.csv")
    assert (
        json.loads(store.open_read(f"{paths.tuning_dir}importances.json").read()) == res.importances
    )
    assert set(res.importances) <= set(SEARCH_SPACE)
    for name in PLOT_NAMES:
        if name in res.plots:
            assert res.plots[name].startswith(b"\x89PNG")
            assert store.exists(f"{paths.tuning_dir}{name}")
    assert "optimization_history.png" in res.plots

    again = run_tune(
        store, ml_run_id="manual__tune", cfg=cfg, registry=registry, n_trials=3, max_rounds=20
    )
    assert again.best_params == res.best_params and again.best_lr_C == res.best_lr_C
    assert again.best_value == res.best_value


def test_the_test_frame_is_refused(store, meta, prepare):
    cfg = small_config()
    res = prepare(store, meta, run_id="manual__tt", cfg=cfg)
    features = feature_list(load_feature_registry(), cfg)
    test = read_frame(store, res.paths, "test")
    with pytest.raises(ValueError, match="test or purge split"):
        tune(test, features, cfg, label=cfg.label, n_trials=1, max_rounds=5)
    mixed = read_frame(store, res.paths, "eval_train").vstack(test.head(1))
    with pytest.raises(ValueError, match="test or purge split"):
        tune(mixed, features, cfg, label=cfg.label, n_trials=1, max_rounds=5)


def test_a_changed_feature_list_needs_a_new_prepare(store, meta, prepare):
    cfg = small_config()
    prepare(store, meta, run_id="manual__tf", cfg=cfg)
    registry = load_feature_registry()
    extra = next(f.name for f in feature_list(registry, cfg) if f.dtype != "str")
    changed = cfg.model_copy(
        update={
            "features": cfg.features.model_copy(update={"exclude": [*cfg.features.exclude, extra]})
        }
    )
    with pytest.raises(MlInputError, match="rerun prepare"):
        run_tune(store, ml_run_id="manual__tf", cfg=changed, registry=registry, n_trials=1)
