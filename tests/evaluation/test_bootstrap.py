"""evaluation/bootstrap.py: the facility block bootstrap (docs/ML_PIPELINE_SPEC.md §6.3)."""

from collections import Counter

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from precursorintelligence.evaluation.bootstrap import facility_bootstrap, facility_resamples


def test_every_replicate_keeps_whole_facilities():
    groups = np.array(["A", "A", "A", "B", "C", "C"])
    sizes = Counter(groups.tolist())
    for idx in facility_resamples(groups, 50, seed=0):
        drawn = Counter(groups[idx].tolist())
        # a facility comes with all its rows, and as many facilities are drawn as there are
        assert all(n % sizes[g] == 0 for g, n in drawn.items())
        assert sum(n // sizes[g] for g, n in drawn.items()) == 3


def test_the_same_seed_gives_the_same_draws_and_another_seed_others():
    groups = np.arange(100)
    a = list(facility_resamples(groups, 5, seed=7))
    b = list(facility_resamples(groups, 5, seed=7))
    c = list(facility_resamples(groups, 5, seed=8))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c, strict=True))


def test_one_row_per_facility_is_the_row_bootstrap():
    groups = np.arange(1000)
    idx = next(facility_resamples(groups, 1, seed=0))
    assert len(idx) == 1000
    assert 0.55 < len(np.unique(idx)) / 1000 < 0.72  # about 1 - 1/e of the rows appear


def test_replicate_count():
    assert len(list(facility_resamples(["x", "y"], 3, seed=0))) == 3


@pytest.mark.parametrize("groups, replicates", [([], 10), ([["a"]], 10), (["a"], 0)])
def test_bad_inputs_are_refused(groups, replicates):
    with pytest.raises(ValueError):
        next(facility_resamples(groups, replicates, seed=0))


def scored(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    risk = rng.beta(2, 5, n)
    y = (rng.random(n) < risk).astype(int)
    noise = rng.normal(0, 0.15, n)
    return y, risk, np.clip(risk + noise, 0, 1)  # a good score and a noisier one


def pr_auc(y, p):
    return {"pr_auc": average_precision_score(y, p), "mean_p": float(np.mean(p))}


def test_estimates_are_on_all_rows_and_the_interval_brackets_them():
    y, good, noisy = scored(3000, seed=1)
    res = facility_bootstrap(
        y, {"good": good, "noisy": noisy}, np.arange(3000), pr_auc, replicates=200, seed=0
    )
    assert res.estimates["good"]["pr_auc"] == pytest.approx(average_precision_score(y, good))
    low, high = res.ci("good", "pr_auc")
    assert low < res.estimates["good"]["pr_auc"] < high
    assert res.replicates == 200 and len(res.draws["noisy"]["mean_p"]) == 200


def test_the_paired_difference_is_tighter_than_the_two_separate_intervals():
    y, good, noisy = scored(3000, seed=2)
    res = facility_bootstrap(
        y, {"good": good, "noisy": noisy}, np.arange(3000), pr_auc, replicates=300, seed=0
    )
    d = res.difference("good", "noisy", "pr_auc")
    assert d["estimate"] == pytest.approx(
        res.estimates["good"]["pr_auc"] - res.estimates["noisy"]["pr_auc"]
    )
    assert d["ci_low"] > 0  # the better score is significantly better
    separate = (res.ci("good", "pr_auc")[1] - res.ci("good", "pr_auc")[0]) + (
        res.ci("noisy", "pr_auc")[1] - res.ci("noisy", "pr_auc")[0]
    )
    assert d["ci_high"] - d["ci_low"] < separate / 2


def test_a_score_compared_with_itself_differs_by_exactly_zero():
    y, good, _ = scored(500, seed=3)
    res = facility_bootstrap(
        y, {"a": good, "b": good.copy()}, np.arange(500), pr_auc, replicates=50, seed=0
    )
    assert res.difference("a", "b", "pr_auc") == {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_draws_with_one_class_are_skipped():
    y = np.array([1, 0, 0, 0])
    res = facility_bootstrap(
        y, {"s": np.array([0.9, 0.1, 0.2, 0.3])}, np.arange(4), pr_auc, replicates=100, seed=0
    )
    assert 0 < res.replicates < 100  # some draws miss the single positive


def test_the_same_seed_gives_the_same_intervals():
    y, good, noisy = scored(800, seed=4)
    run = lambda: facility_bootstrap(  # noqa: E731
        y, {"g": good}, np.arange(800), pr_auc, replicates=50, seed=9
    ).ci("g", "pr_auc")
    assert run() == run()


@pytest.mark.parametrize(
    "scores, groups",
    [({}, np.arange(3)), ({"s": [0.1, 0.2]}, np.arange(3)), ({"s": [0.1, 0.2, 0.3]}, np.arange(2))],
)
def test_mismatched_inputs_are_refused(scores, groups):
    with pytest.raises(ValueError):
        facility_bootstrap([0, 1, 0], scores, groups, pr_auc, replicates=5, seed=0)
