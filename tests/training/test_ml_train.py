"""`ml train` on the synthetic gold (docs/ML_PIPELINE_SPEC.md §3.3, §4, §5, §14, §17.1)."""

import io
import json
import sys

import numpy as np
import polars as pl
import pytest
import yaml
from make_ml_gold import ml_config

from precursorintelligence.common.config import load_feature_registry
from precursorintelligence.harmonize.cli import main
from precursorintelligence.training import MlInputError
from precursorintelligence.training.contracts import (
    CANDIDATES,
    OOF_KEYS,
    RunPaths,
    Selection,
    TrainResult,
    oof_columns,
)
from precursorintelligence.training.data import read_frame, read_windows, to_model_frame
from precursorintelligence.training.ensemble import read_package
from precursorintelligence.training.select import BiasOutcome
from precursorintelligence.training.train import psi, run_train


def small_config(**members):
    """The fixture-sized ml config with small LightGBM fits (fast tests)."""
    cfg = ml_config()
    lgbm = {**cfg.members.lgbm, "n_estimators": 40, "min_child_samples": 20, "num_threads": 1}
    lgbm.update(members)
    return cfg.model_copy(update={"members": cfg.members.model_copy(update={"lgbm": lgbm})})


def _read(store, key):
    with store.open_read(key) as f:
        return f.read()


@pytest.fixture
def trained(store, meta, prepare, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = small_config()
    prepare(store, meta, run_id="manual__train", cfg=cfg)
    out = run_train(
        store,
        ml_run_id="manual__train",
        cfg=cfg,
        registry=load_feature_registry(),
        recipe_hash="test",
    )
    return cfg, out


def test_train_writes_every_file_of_its_contract(store, trained):
    cfg, out = trained
    paths = RunPaths.of(cfg, "manual__train")
    windows = read_windows(store, paths)
    result = TrainResult.model_validate_json(_read(store, paths.train_result))
    assert result == out.result
    assert Selection.model_validate_json(_read(store, paths.selection)) == out.selection
    assert set(result.eval_packages) == set(CANDIDATES)
    for c, prefix in result.eval_packages.items():
        assert prefix == paths.package("eval", c)
        model = read_package(store, prefix)
        assert model.meta is not None and model.meta.candidate == c
        assert model.meta.training_months == {"eval": windows.eval.months}
    assert result.refit_package == paths.package("refit", result.selected)
    refit = read_package(store, result.refit_package)
    assert refit.meta is not None and list(refit.meta.training_months) == ["refit"]
    eval_oof = pl.read_parquet(io.BytesIO(_read(store, paths.oof("eval"))))
    expected = list(OOF_KEYS) + [col for c in cfg.selection.candidates for col in oof_columns(c)]
    assert eval_oof.columns == expected
    refit_oof = pl.read_parquet(io.BytesIO(_read(store, paths.oof("refit"))))
    assert refit_oof.columns == list(OOF_KEYS) + list(oof_columns(result.selected))
    assert set(result.calibration) == {f"eval/{c}" for c in CANDIDATES} | {
        f"refit/{result.selected}"
    }
    assert result.mlflow_run_id is None and result.refit_model_uri == ""
    assert result.seconds["total"] > 0
    assert {p.feature for p in result.psi} <= set(cfg.features.regime_sensitive)


def test_the_eval_packages_score_like_the_oof_columns_say(store, trained):
    """Package predictions are the calibrated ensemble of the members' log-odds."""
    cfg, out = trained
    paths = RunPaths.of(cfg, "manual__train")
    frame = read_frame(store, paths, "eval_train")
    for c, model in out.eval_models.items():
        pred = model.predict_frame(frame.head(200))
        assert np.allclose(pred["p"], 1 / (1 + np.exp(-pred["logit"])))
        platt = out.result.calibration[f"eval/{c}"]
        assert np.allclose(pred["logit"], platt.a * pred["logit_raw"] + platt.b)
        if c == "ensemble":
            w = cfg.ensemble.weights
            mean = (w["lgbm"] * pred["logit_lgbm"] + w["lr"] * pred["logit_lr"]) / sum(w.values())
            assert np.allclose(pred["logit_raw"], mean)


def test_platt_is_fitted_on_the_newest_months_and_oof_ranks_better_than_chance(store, trained):
    cfg, out = trained
    paths = RunPaths.of(cfg, "manual__train")
    frame = read_frame(store, paths, "eval_train")
    newest = sorted(frame["t"].unique().to_list())[-cfg.calibration.fit_recent_months :]
    for key, platt in out.result.calibration.items():
        if key.startswith("eval/"):
            assert platt.fit_months == newest
            assert platt.fit_rows == frame.filter(pl.col("t").is_in(newest)).height
    best = max(r.oof_pr_auc for r in out.selection.candidates)
    base = float(frame["y_g12"].mean())
    assert best > base + 0.05  # the fixture plants a signal


def test_the_selection_uses_the_bias_check(store, meta, prepare, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = small_config()
    prepare(store, meta, run_id="manual__bias", cfg=cfg)
    seen = []

    def only_lr_passes(rows, p, cfg_):
        seen.append(len(p))
        return BiasOutcome(passed=len(seen) == 3, detail={"call": len(seen)})  # 3rd candidate = lr

    out = run_train(
        store,
        ml_run_id="manual__bias",
        cfg=cfg,
        registry=load_feature_registry(),
        bias_check=only_lr_passes,
        recipe_hash="test",
    )
    assert cfg.selection.candidates[2] == "lr"
    assert out.selection.selected == "lr" and out.result.selected == "lr"
    assert "bias_check_failed:ensemble" in out.result.flags
    assert set(seen) == {read_frame(store, RunPaths.of(cfg, "manual__bias"), "eval_train").height}


def test_train_refuses_a_feature_list_that_changed_since_prepare(store, meta, prepare):
    cfg = small_config()
    prepare(store, meta, run_id="manual__chg", cfg=cfg)
    changed = cfg.model_copy(
        update={
            "features": cfg.features.model_copy(
                update={"exclude": [*cfg.features.exclude, "hprd_total"]}
            )
        }
    )
    with pytest.raises(MlInputError, match="rerun prepare"):
        run_train(
            store,
            ml_run_id="manual__chg",
            cfg=changed,
            registry=load_feature_registry(),
            recipe_hash="test",
        )


def test_train_is_deterministic(store, meta, prepare, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = small_config()
    prepare(store, meta, run_id="manual__det", cfg=cfg)
    reg = load_feature_registry()
    a = run_train(store, ml_run_id="manual__det", cfg=cfg, registry=reg, recipe_hash="test")
    b = run_train(store, ml_run_id="manual__det", cfg=cfg, registry=reg, recipe_hash="test")
    for c in CANDIDATES:
        assert a.eval_models[c].to_files() == b.eval_models[c].to_files()
    assert a.selection == b.selection


def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    x = pl.Series(rng.normal(size=5000))
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)
    assert psi(x, pl.Series(rng.normal(loc=2.0, size=5000))) > 0.25
    cat = pl.Series(["a"] * 50 + ["b"] * 50)
    assert psi(cat, pl.Series(["a"] * 90 + ["b"] * 10), categorical=True) > 0.25
    with_nulls = pl.Series([1.0, None, 2.0, None])
    assert psi(with_nulls, pl.Series([1.0, 2.0, 1.0, 2.0])) > 0  # nulls are their own bin


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["precursor-pipeline", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


def test_ml_train_command_runs_on_a_prepared_folder(gold, monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg_path = tmp_path / "ml.yaml"
    cfg_path.write_text(yaml.safe_dump(small_config().model_dump(mode="json"), sort_keys=False))
    common = [
        "--env-file",
        str(gold.env_file),
        "--run-id",
        "manual__cli",
        "--ml-config",
        str(cfg_path),
    ]
    assert _run(monkeypatch, "ml", "prepare", *common) == 0
    capsys.readouterr()
    code = _run(monkeypatch, "ml", "train", *common)
    out = capsys.readouterr().out
    assert code == 0, out
    summary = json.loads(out.strip().splitlines()[-1])
    assert summary["ml_run_id"] == "manual__cli" and summary["selected"] in CANDIDATES
    assert (gold.root / "ml/runs/manual__cli/train.json").is_file()


def test_ml_train_without_a_prepared_folder_is_an_error(gold, monkeypatch):
    code = _run(monkeypatch, "ml", "train", "--env-file", str(gold.env_file), "--run-id", "nothing")
    assert code == 1


def test_the_model_frame_of_the_refit_package_matches_to_model_frame(store, trained):
    cfg, out = trained
    paths = RunPaths.of(cfg, "manual__train")
    score = read_frame(store, paths, "score")
    x = to_model_frame(score, out.refit_model.features)
    assert list(x.columns) == out.result.features
    assert np.allclose(
        out.refit_model.predict_frame(score)["p"], out.refit_model.predict(None, x)["p"]
    )
