"""training/tune.py on the synthetic gold: the proposal, the files, determinism, never the test frame
(docs/ML_PIPELINE_SPEC.md §13, §8.3, §20)."""

import json
import sys

import pytest
import yaml
from make_ml_gold import ml_config

from precursorintelligence.common.config import TuningParamCfg, load_feature_registry
from precursorintelligence.harmonize.cli import main
from precursorintelligence.ingestion.pipeline import EXIT_CONFIG, EXIT_OK, EXIT_VALIDATION
from precursorintelligence.training import MlInputError
from precursorintelligence.training import tune as tune_mod
from precursorintelligence.training.contracts import RunPaths
from precursorintelligence.training.data import feature_list, read_frame
from precursorintelligence.training.tune import (
    LR_C,
    PLOT_NAMES,
    run_tune,
    search_space,
    tune,
)


def small_config(**tuning):
    cfg = ml_config()
    t = cfg.tuning.model_copy(
        update={"timeout_minutes": 2.0, "early_stopping_rounds": 10, "max_rounds": 20, **tuning}
    )
    return cfg.model_copy(update={"tuning": t})


def test_the_proposal_is_in_the_space_and_written(store, meta, prepare):
    cfg = small_config()
    prepare(store, meta, run_id="manual__tune", cfg=cfg)
    registry = load_feature_registry()
    res = run_tune(
        store, ml_run_id="manual__tune", cfg=cfg, registry=registry, n_trials=3, max_rounds=20
    )

    proposal = yaml.safe_load(res.proposal_yaml)
    lgbm = proposal["members"]["lgbm"]
    space = search_space(cfg)
    for name, p in space.items():
        if name == LR_C:
            continue
        assert p.low <= lgbm[name] <= p.high, name
    for fixed in ("seed", "deterministic", "force_col_wise", "num_threads", "subsample_freq"):
        assert lgbm[fixed] == getattr(cfg.members.lgbm, fixed)
    assert type(cfg.members.lgbm).model_validate(lgbm)  # the proposal is a valid members.lgbm
    assert isinstance(lgbm["n_estimators"], int) and 1 <= lgbm["n_estimators"] <= 20
    assert lgbm == res.best_params
    assert proposal["members"]["lr"]["C"] == res.best_lr_C in space[LR_C].choices
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
    assert set(res.importances) <= set(space)
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
    with pytest.raises(
        MlInputError, match="test or purge split"
    ):  # a data refusal: exit 2, no retry
        tune(test, features, cfg, label=cfg.label, n_trials=1, max_rounds=5)
    mixed = read_frame(store, res.paths, "eval_train").vstack(test.head(1))
    with pytest.raises(MlInputError, match="test or purge split"):
        tune(mixed, features, cfg, label=cfg.label, n_trials=1, max_rounds=5)
    with pytest.raises(MlInputError, match="split_g12"):
        tune(test.drop("split_g12"), features, cfg, label=cfg.label, n_trials=1, max_rounds=5)


def test_the_round_cap_comes_from_tuning_max_rounds(store, meta, prepare, monkeypatch):
    cfg = small_config(max_rounds=7)
    prepare(store, meta, run_id="manual__cap", cfg=cfg)
    seen: list[int] = []
    real_fit = tune_mod.LgbmMember.fit

    def spy(x, y, features, params, **kw):
        seen.append(params.n_estimators)
        return real_fit(x, y, features, params, **kw)

    monkeypatch.setattr(tune_mod.LgbmMember, "fit", spy)
    res = run_tune(
        store, ml_run_id="manual__cap", cfg=cfg, registry=load_feature_registry(), n_trials=1
    )
    assert seen and set(seen) == {7}
    assert 1 <= res.best_params["n_estimators"] <= 7


def test_the_space_comes_from_the_config(store, meta, prepare):
    """A trimmed tuning.space searches only its keys; the rest keep members.lgbm, and LR keeps members.lr.C."""
    space = {"learning_rate": TuningParamCfg(kind="float", low=0.05, high=0.06)}
    cfg = small_config(space=space)
    prepare(store, meta, run_id="manual__space", cfg=cfg)
    res = run_tune(
        store, ml_run_id="manual__space", cfg=cfg, registry=load_feature_registry(), n_trials=2
    )
    assert {c.removeprefix("params_") for c in res.trials.columns if c.startswith("params_")} == {
        "learning_rate"
    }
    assert 0.05 <= res.best_params["learning_rate"] <= 0.06
    fixed = cfg.members.lgbm.as_dict()
    for k, v in res.best_params.items():
        if k not in ("learning_rate", "n_estimators"):
            assert v == fixed[k], k
    assert res.best_lr_C == cfg.members.lr.C


# ------------------------------------------------------------------ `ml tune` exit codes


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["precursor-pipeline", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


def _write_cfg(tmp_path, cfg, **lgbm_extra):
    data = cfg.model_dump(mode="json")
    data["members"]["lgbm"] = {k: v for k, v in data["members"]["lgbm"].items() if v is not None}
    data["members"]["lgbm"].update(lgbm_extra)
    path = tmp_path / "ml.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_ml_tune_command_runs_and_exits_0(gold, monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    common = ["--env-file", str(gold.env_file), "--run-id", "manual__ct"]
    common += ["--ml-config", str(_write_cfg(tmp_path, small_config()))]
    assert _run(monkeypatch, "ml", "prepare", *common) == EXIT_OK
    capsys.readouterr()
    code = _run(monkeypatch, "ml", "tune", *common, "--n-trials", "1")
    out = capsys.readouterr().out
    assert code == EXIT_OK, out
    summary = json.loads(out.strip().splitlines()[-1])
    assert summary["ml_run_id"] == "manual__ct" and summary["n_trials"] == 1


def test_ml_tune_refusal_exits_2_not_1(gold, monkeypatch, capsys, tmp_path):
    """A data refusal must exit 2 (no Airflow retry), not 1."""

    def refuse(*args, **kwargs):
        raise MlInputError("tune refuses rows of the test or purge split")

    monkeypatch.setattr(tune_mod, "run_tune", refuse)
    code = _run(monkeypatch, "ml", "tune", "--env-file", str(gold.env_file), "--run-id", "x")
    assert code == EXIT_VALIDATION
    assert "refused" in capsys.readouterr().err


def test_ml_tune_with_an_unknown_lgbm_key_exits_3(gold, monkeypatch, capsys, tmp_path):
    path = _write_cfg(tmp_path, small_config(), num_leave=3)  # a typo of num_leaves
    code = _run(
        monkeypatch,
        "ml",
        "tune",
        "--env-file",
        str(gold.env_file),
        "--run-id",
        "x",
        "--ml-config",
        str(path),
    )
    assert code == EXIT_CONFIG
    assert "num_leave" in capsys.readouterr().err


def test_ml_tune_without_a_prepared_folder_is_an_error(gold, monkeypatch):
    code = _run(monkeypatch, "ml", "tune", "--env-file", str(gold.env_file), "--run-id", "nothing")
    assert code == 1


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
