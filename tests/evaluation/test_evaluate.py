"""evaluation/evaluate.py on a trained synthetic run (docs/ML_PIPELINE_SPEC.md §6, §7, §17.1 leakage tests 1, 3, 10)."""

import hashlib
import io
import json

import numpy as np
import polars as pl
import pytest
from conftest import FakeRegistry

from precursorintelligence.evaluation import metrics
from precursorintelligence.evaluation.evaluate import (
    CHAMPION,
    RULE,
    check_training_months,
    evaluate_run,
    report_prefix,
    run_evaluate,
)
from precursorintelligence.evaluation.report import EvaluationReport, comparison_key
from precursorintelligence.training import MlInputError
from precursorintelligence.training.contracts import RunPaths
from precursorintelligence.training.data import read_frame, read_windows
from precursorintelligence.training.ensemble import read_package


def paths(run):
    return RunPaths.of(run.cfg, run.run_id)


def evaluate(run, **kw):
    kw.setdefault("shuffled_label", False)
    return evaluate_run(run.store(), ml_run_id=run.run_id, cfg=run.cfg, **kw)


@pytest.fixture
def evaluated(trained):
    out = run_evaluate(
        trained.store(), ml_run_id=trained.run_id, cfg=trained.cfg, shuffled_label=False
    )
    return trained, out


def test_evaluate_writes_the_report_and_its_files(evaluated):
    run, _ = evaluated
    store, p = run.store(), paths(run)
    with store.open_read(p.evaluation) as f:
        data = f.read()
    report = EvaluationReport.model_validate_json(data)
    assert report.selected == run.train.result.selected
    assert set(report.candidates) == set(run.train.result.eval_packages)
    assert set(report.baselines) == {"prior_gplus_rule", "severity_24m", "star_rating"}
    assert b"NaN" not in data  # JSON has no NaN: undefined numbers are null
    for name in (
        "slices.csv",
        "shap_mean_abs.csv",
        "per_month.csv",
        "pr_curve.png",
        "reliability.png",
    ):
        assert store.exists(f"{report_prefix(p)}{name}"), name


def test_the_selected_candidates_metrics_are_its_eval_fit_on_the_test_frame(evaluated):
    run, out = evaluated
    store, p = run.store(), paths(run)
    test = read_frame(store, p, "test")
    chosen = run.train.result.selected
    prob = read_package(store, run.train.result.eval_packages[chosen]).predict_frame(test)["p"]
    y = test[f"y_{run.cfg.label}"].to_numpy()
    r = out.report.candidates[chosen]
    assert r.ranking["pr_auc"] == pytest.approx(metrics.ranking_metrics(y, prob, [0.1])["pr_auc"])
    assert r.probability["ece"] == pytest.approx(metrics.ece(y, prob, run.cfg.evaluation.ece_bins))
    iv = r.intervals["pr_auc"]
    assert iv.ci_low <= iv.estimate <= iv.ci_high
    assert out.report.rows == test.height and out.report.positives == int(y.sum())


def test_the_threshold_comes_from_the_oof_predictions(evaluated):
    run, out = evaluated
    oof = pl.read_parquet(run.gold.root / run.train.result.oof["eval"])
    chosen = run.train.result.selected
    assert out.report.threshold == metrics.max_f1_threshold(oof["y"], oof[f"p_{chosen}"])
    assert out.report.candidates[chosen].classification["threshold"] == out.report.threshold


def test_g3s_comparison_with_the_rule_is_paired(evaluated):
    run, out = evaluated
    chosen = run.train.result.selected
    iv = out.report.comparison("pr_auc", chosen, RULE)
    model_pr = out.report.candidates[chosen].ranking["pr_auc"]
    assert iv.estimate == pytest.approx(model_pr - out.report.baselines[RULE].ranking["pr_auc"])
    assert iv.ci_low <= iv.estimate <= iv.ci_high


def test_slices_and_sanity_are_reported(evaluated):
    _, out = evaluated
    r = out.report
    assert {"ownership_group", "bed_band", "state"} <= {s["slice"] for s in r.slices}
    assert r.sanity.shap_additive is True and r.sanity.shuffled_label_roc is None
    assert r.champion is None and r.slices_vs_champion is None
    assert r.top_features and r.top_features[0]["mean_abs"] >= r.top_features[-1]["mean_abs"]


def _replace_frame(run, name, df):
    """Write a frame and re-pin its hash in the input manifest (read_frame checks it)."""
    store, p = run.store(), paths(run)
    buf = io.BytesIO()
    df.write_parquet(buf)
    data = buf.getvalue()
    store.put_bytes(p.frame(name), data, overwrite=True)
    with store.open_read(p.input_manifest) as f:
        manifest = json.loads(f.read())
    manifest["frames_sha256"][name] = hashlib.sha256(data).hexdigest()
    store.put_bytes(p.input_manifest, json.dumps(manifest).encode(), overwrite=True)


def test_permuting_the_test_labels_leaves_the_threshold_alone(trained):
    # leakage test 1 for evaluate: the cut-off is chosen on OOF predictions, never on test labels
    before = evaluate(trained)
    test = read_frame(trained.store(), paths(trained), "test")
    label = f"y_{trained.cfg.label}"
    permuted = np.random.default_rng(0).permutation(test[label].to_numpy())
    _replace_frame(trained, "test", test.with_columns(pl.Series(label, permuted)))
    after = evaluate(trained)
    chosen = trained.train.result.selected
    assert after.report.threshold == before.report.threshold
    assert after.report.candidates[chosen].ranking["pr_auc"] != pytest.approx(
        before.report.candidates[chosen].ranking["pr_auc"]
    )


def test_a_package_trained_into_the_test_window_is_refused(trained):
    # leakage test 3
    store, p = trained.store(), paths(trained)
    windows = read_windows(store, p)
    model = read_package(store, trained.train.result.eval_packages["lr"])
    check_training_months(model, "lr", windows)  # the real one passes
    last = max(model.meta.training_months["eval"])
    late = windows.model_copy(update={"test": windows.test.model_copy(update={"months": [last]})})
    with pytest.raises(MlInputError, match="reaches the test window"):
        check_training_months(model, "lr", late)


def test_the_champions_eval_fit_is_compared_on_the_same_rows(trained):
    chosen = trained.train.result.selected
    same = read_package(trained.store(), trained.train.result.eval_packages[chosen])
    reg = FakeRegistry({"models:/champ-eval": same})
    reg.register(
        "models:/champ-refit", {"eval_model_uri": "models:/champ-eval", "candidate": chosen}
    )
    reg.set_alias("champion", 1)
    r = evaluate(trained, registry=reg).report
    assert r.champion.version == 1 and r.champion_eval is not None
    for m in ("pr_auc", "brier"):
        iv = r.comparison(m, chosen, CHAMPION)
        assert (iv.estimate, iv.ci_low, iv.ci_high) == (0, 0, 0)  # the same model: no difference
    assert comparison_key("pr_auc", chosen, CHAMPION) in r.comparisons
    assert r.slices_vs_champion is not None
    assert all(s.abs_oe_diff.estimate == 0 for s in r.slices_vs_champion)


def test_a_champion_without_its_eval_fit_is_refused(trained):
    reg = FakeRegistry()
    reg.register("models:/x", {"candidate": "lgbm"})
    reg.set_alias("champion", 1)
    with pytest.raises(MlInputError, match="eval_model_uri"):
        evaluate(trained, registry=reg)


def test_shuffled_labels_run_only_for_a_recipe_no_gate_judged(trained):
    meta = trained.meta()
    first = evaluate(trained, meta=meta, shuffled_label=None)
    assert first.report.sanity.shuffled_label_roc is not None
    meta.append("ml_runs", [{"ml_run_id": "earlier", "action": "hold", "recipe_hash": "test"}])
    again = evaluate(trained, meta=meta, shuffled_label=None)
    assert again.report.sanity.shuffled_label_roc is None


def test_evaluate_needs_a_trained_run(trained):
    (trained.gold.root / paths(trained).train_result).unlink()
    with pytest.raises(MlInputError, match="ml train"):
        evaluate(trained)
