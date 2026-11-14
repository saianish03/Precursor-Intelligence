"""evaluation/sanity.py: the shuffled-label and SHAP-additivity checks (docs/ML_PIPELINE_SPEC.md §6.4)."""

import numpy as np
import pytest

from precursorintelligence.common.config import load_feature_registry
from precursorintelligence.evaluation.sanity import (
    shap_additivity,
    shuffle_within_groups,
    shuffled_label_roc,
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
