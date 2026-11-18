"""evaluation/sanity.py: the shuffled-label and SHAP-additivity checks (docs/ML_PIPELINE_SPEC.md §6.4)."""

import numpy as np
import pytest

from precursorintelligence.common.config import load_feature_registry
from precursorintelligence.evaluation import sanity
from precursorintelligence.evaluation.gates import g2_sanity
from precursorintelligence.evaluation.sanity import (
    shap_additivity,
    shuffle_within_groups,
    shuffled_label_roc,
    shuffled_label_rocs,
)
from precursorintelligence.training.contracts import RunPaths
from precursorintelligence.training.data import read_frame, read_input_manifest
from precursorintelligence.training.ensemble import read_package
from precursorintelligence.training.train import resolve_features


def test_shuffling_keeps_each_months_labels():
    y = np.array([1, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    months = np.array(["a"] * 4 + ["b"] * 6)
    out = shuffle_within_groups(y, months, seed=0)
    for m in ("a", "b"):
        assert sorted(out[months == m]) == sorted(y[months == m])
    assert np.array_equal(shuffle_within_groups(y, months, seed=0), out)  # seeded


def test_shuffling_breaks_the_link_to_the_rows():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    out = shuffle_within_groups(y, np.zeros(5000), seed=2)
    assert abs(np.corrcoef(y, out)[0, 1]) < 0.05


def frames(run):
    store, paths = run.store(), RunPaths.of(run.cfg, run.run_id)
    features = resolve_features(load_feature_registry(), run.cfg, read_input_manifest(store, paths))
    return store, paths, features


def test_a_model_on_shuffled_labels_ranks_like_chance(trained):
    store, paths, features = frames(trained)
    train, test = read_frame(store, paths, "eval_train"), read_frame(store, paths, "test")
    roc = shuffled_label_roc(train, test, features, trained.cfg)
    assert 0.40 < roc < 0.60  # a few hundred test rows: wider than production's 0.48-0.52


def test_g2_averages_several_shuffles_each_with_its_own_seed(trained):
    # one shuffle is too noisy for G2's range (#74 B20): every shuffle's value is kept
    store, paths, features = frames(trained)
    train, test = read_frame(store, paths, "eval_train"), read_frame(store, paths, "test")
    rocs = shuffled_label_rocs(train, test, features, trained.cfg)
    assert len(rocs) == trained.cfg.evaluation.shuffled_label.shuffles > 1
    assert len(set(rocs)) == len(rocs)  # different seeds, different shuffles
    assert rocs[0] == shuffled_label_roc(train, test, features, trained.cfg, seed=0)
    assert 0.40 < sum(rocs) / len(rocs) < 0.60


def test_labels_that_are_not_really_shuffled_still_fail_g2(trained, monkeypatch):
    # what the check exists for: a model that still sees the real labels ranks far above chance
    monkeypatch.setattr(sanity, "shuffle_within_groups", lambda y, groups, seed: y)
    store, paths, features = frames(trained)
    train, test = read_frame(store, paths, "eval_train"), read_frame(store, paths, "test")
    rocs = shuffled_label_rocs(train, test, features, trained.cfg)
    g2 = g2_sanity(
        uncleared_leakage_flags=[],
        shap_additive=True,
        shuffled_label_roc=sum(rocs) / len(rocs),
        shuffled_label_rocs=rocs,
        roc_range=trained.cfg.evaluation.shuffled_label.roc_range,
    )
    assert g2.status == "fail" and "outside" in g2.reason


def test_shap_contributions_add_up_for_every_candidate(trained):
    store, paths, _ = frames(trained)
    test = read_frame(store, paths, "test")
    for prefix in trained.train.result.eval_packages.values():
        check = shap_additivity(read_package(store, prefix), test, rows=200)
        assert check.holds, check
        assert check.rows == min(200, test.height)


def test_additivity_needs_rows(trained):
    store, paths, _ = frames(trained)
    model = read_package(store, trained.train.result.refit_package)
    with pytest.raises(ValueError, match="no rows"):
        shap_additivity(model, read_frame(store, paths, "test").clear())
