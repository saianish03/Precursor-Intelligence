"""evaluation/evaluate.py on a trained synthetic run (docs/ML_PIPELINE_SPEC.md §6, §7, §17.1 leakage tests 1, 3, 10)."""

import copy
import hashlib
import io
import json

import numpy as np
import polars as pl
import pytest
from ml_eval_fixtures import FakeRegistry

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
from precursorintelligence.gold.splits import shift
from precursorintelligence.training import MlInputError
from precursorintelligence.training.contracts import RunPaths, TrainResult
from precursorintelligence.training.data import read_frame, read_windows
from precursorintelligence.training.ensemble import (
    PACKAGE_META,
    PrecursorModel,
    read_package,
    write_package,
)


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
        "bias_slices.png",
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
    # the mitigation trade-off is on OOF predictions: the test labels never reach it (equal up to float sums)
    for a, b in zip(
        after.report.mitigation.variants, before.report.mitigation.variants, strict=True
    ):
        assert a.model_dump() == pytest.approx(b.model_dump(), abs=1e-9)
    assert after.report.candidates[chosen].ranking["pr_auc"] != pytest.approx(
        before.report.candidates[chosen].ranking["pr_auc"]
    )


def test_leakage_test_3_sits_exactly_at_the_horizon(trained):
    # trained on L: a test window starting at L + H is refused (an outcome window reaches it), L + H + 1 is not
    store, p = trained.store(), paths(trained)
    windows = read_windows(store, p)
    model = read_package(store, trained.train.result.eval_packages["lr"])
    check_training_months(model, "lr", windows)  # the real one passes
    last, h = max(model.meta.training_months["eval"]), windows.horizon_months
    with pytest.raises(MlInputError, match="reaches the test window"):
        check_training_months(model, "lr", windows, shift(last, h))
    check_training_months(model, "lr", windows, shift(last, h + 1))


def test_the_refit_package_is_never_scored_on_test(trained):
    # leakage test 3 end to end: train.json pointing evaluate at the refit (trained up to the test months)
    store, p = trained.store(), paths(trained)
    with store.open_read(p.train_result) as f:
        train = TrainResult.model_validate_json(f.read())
    chosen = train.selected
    bad = train.model_copy(
        update={"eval_packages": {**train.eval_packages, chosen: train.refit_package}}
    )
    store.put_bytes(p.train_result, bad.model_dump_json().encode(), overwrite=True)
    with pytest.raises(MlInputError, match="not an eval fit"):
        evaluate(trained)


SOURCE = "champ-run"  # the run whose gate registered the champion (version 1)


def champion_from(run, model, *, record=True):
    """Version 1 is @champion, registered by SOURCE's gate (an ml_runs hold row) and, with ``record``, audited by
    a manual_promote row. SOURCE's run folder holds ``model`` as its selected candidate's eval fit, with the sha256
    its train.json recorded. The fake registry holds no model: MLflow's copy must never be loaded."""
    store, train = run.store(), run.train.result
    sp = RunPaths.of(run.cfg, SOURCE)
    prefix = sp.package("eval", train.selected)
    sha = write_package(store, prefix, model)
    source_train = train.model_copy(
        update={
            "ml_run_id": SOURCE,
            "eval_packages": {train.selected: prefix},
            "eval_package_sha256": {train.selected: sha},
        }
    )
    store.put_bytes(sp.train_result, source_train.model_dump_json().encode(), overwrite=True)
    meta = run.meta()
    meta.append("ml_runs", [{"ml_run_id": SOURCE, "action": "hold", "candidate_version": 1}])
    if record:
        meta.append(
            "ml_runs",
            [{"ml_run_id": SOURCE, "action": "manual_promote", "champion_version_after": 1}],
        )
    reg = FakeRegistry()
    reg.register(
        "models:/champ-refit", {"eval_model_uri": "models:/champ-eval", "candidate": train.selected}
    )
    reg.set_alias("champion", 1)
    return reg, meta, sp, prefix, source_train


def selected_eval_fit(run):
    return read_package(run.store(), run.train.result.eval_packages[run.train.result.selected])


def rewrite_source_train(run, sp, **update):
    store = run.store()
    with store.open_read(sp.train_result) as f:
        train = TrainResult.model_validate_json(f.read())
    store.put_bytes(
        sp.train_result, train.model_copy(update=update).model_dump_json().encode(), overwrite=True
    )


def test_train_records_the_sha256_of_every_eval_package_file(trained):
    store, train = trained.store(), trained.train.result
    assert set(train.eval_package_sha256) == set(train.eval_packages)
    for c, files in train.eval_package_sha256.items():
        assert PACKAGE_META in files
        for name, digest in files.items():
            with store.open_read(f"{train.eval_packages[c]}{name}") as f:
                assert hashlib.sha256(f.read()).hexdigest() == digest


def test_a_champion_trained_into_the_test_window_is_refused(trained):
    # a wrongly recorded eval fit would win G5 unfairly: leakage test 3 holds for the champion too
    store, p = trained.store(), paths(trained)
    model = selected_eval_fit(trained)
    test_month = min(read_windows(store, p).test.months)
    leaky = copy.copy(model)
    leaky.meta = model.meta.model_copy(update={"training_months": {"eval": [test_month]}})
    reg, meta, *_ = champion_from(trained, leaky)
    with pytest.raises(MlInputError, match="champion trained on"):
        evaluate(trained, registry=reg, meta=meta)


def test_a_champion_alias_moved_outside_the_gates_is_refused(trained):
    # §9.4: the alias must be what the latest promotion row in ml_runs recorded, before anything is loaded
    reg, meta, *_ = champion_from(trained, selected_eval_fit(trained), record=False)
    with pytest.raises(MlInputError, match="moved outside gate/promote"):
        evaluate(trained, registry=reg, meta=meta)
    meta.append(
        "ml_runs", [{"ml_run_id": SOURCE, "action": "manual_promote", "champion_version_after": 1}]
    )
    assert evaluate(trained, registry=reg, meta=meta).report.champion


def test_a_tampered_champion_file_is_refused_before_loading(trained, monkeypatch):
    reg, meta, _, prefix, _ = champion_from(trained, selected_eval_fit(trained))
    store = trained.store()
    with store.open_read(f"{prefix}{PACKAGE_META}") as f:
        data = f.read()
    store.put_bytes(f"{prefix}{PACKAGE_META}", data + b" ", overwrite=True)
    real = PrecursorModel.from_files

    def guarded(files):
        if files.get(PACKAGE_META, b"").endswith(b" "):
            raise AssertionError("the tampered package was unpickled before its sha256 was checked")
        return real(files)

    monkeypatch.setattr(PrecursorModel, "from_files", staticmethod(guarded))
    with pytest.raises(MlInputError, match="does not match the sha256"):
        evaluate(trained, registry=reg, meta=meta)


def test_a_champion_without_recorded_hashes_is_refused(trained):
    reg, meta, sp, _, _ = champion_from(trained, selected_eval_fit(trained))
    rewrite_source_train(trained, sp, eval_package_sha256={})  # a train.json from before the hashes
    with pytest.raises(MlInputError, match="records no eval package hashes"):
        evaluate(trained, registry=reg, meta=meta)


def test_a_champion_package_outside_its_run_folder_is_refused(trained):
    reg, meta, sp, _, source_train = champion_from(trained, selected_eval_fit(trained))
    chosen = source_train.selected
    elsewhere = trained.train.result.eval_packages[chosen]  # another run's folder; its hashes match
    rewrite_source_train(trained, sp, eval_packages={chosen: elsewhere})
    with pytest.raises(MlInputError, match="outside its run folder"):
        evaluate(trained, registry=reg, meta=meta)


def test_a_missing_champion_file_is_refused(trained):
    reg, meta, _, prefix, _ = champion_from(trained, selected_eval_fit(trained))
    (trained.gold.root / f"{prefix}{PACKAGE_META}").unlink()
    with pytest.raises(MlInputError, match="is missing"):
        evaluate(trained, registry=reg, meta=meta)


def test_a_champion_without_ml_runs_is_refused(trained):
    reg, *_ = champion_from(trained, selected_eval_fit(trained))
    with pytest.raises(MlInputError, match="without ml_runs"):
        evaluate(trained, registry=reg)


def test_a_champion_no_gate_registered_is_refused(trained):
    reg, meta, *_ = champion_from(trained, selected_eval_fit(trained))
    reg.register("models:/other", {"eval_model_uri": "models:/other-eval"})
    reg.set_alias("champion", 2)
    meta.append(
        "ml_runs", [{"ml_run_id": "x", "action": "manual_promote", "champion_version_after": 2}]
    )
    with pytest.raises(MlInputError, match="no gate run that registered version 2"):
        evaluate(trained, registry=reg, meta=meta)


def test_the_champions_eval_fit_is_compared_on_the_same_rows(trained):
    chosen = trained.train.result.selected
    reg, meta, *_ = champion_from(trained, selected_eval_fit(trained))
    r = evaluate(trained, registry=reg, meta=meta).report
    assert r.champion.version == 1 and r.champion_eval is not None
    for m in ("pr_auc", "brier"):
        iv = r.comparison(m, chosen, CHAMPION)
        assert (iv.estimate, iv.ci_low, iv.ci_high) == (0, 0, 0)  # the same model: no difference
    assert comparison_key("pr_auc", chosen, CHAMPION) in r.comparisons
    assert r.slices_vs_champion is not None
    assert all(s.abs_oe_diff.estimate == 0 for s in r.slices_vs_champion)


def test_a_champion_without_the_eval_model_uri_tag_is_still_compared(trained):
    # the tag is reported only since #66: the eval fit comes verified from GCS
    reg, meta, *_ = champion_from(trained, selected_eval_fit(trained))
    reg.versions[1].pop("eval_model_uri")
    r = evaluate(trained, registry=reg, meta=meta).report
    assert (
        r.champion.version == 1 and r.champion.eval_model_uri == "" and r.champion_eval is not None
    )


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
