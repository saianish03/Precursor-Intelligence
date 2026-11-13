"""evaluation/metrics.py: the §6.1 metrics, checked against scikit-learn where it has the same metric."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from precursorintelligence.evaluation.metrics import (
    calibration_line,
    ece,
    pct,
    probability_metrics,
    ranking_metrics,
    reliability_table,
    top_k,
)


def scored(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn from calibrated probabilities: p is the true risk of each row."""
    rng = np.random.default_rng(seed)
    p = rng.beta(2, 5, n)
    y = (rng.random(n) < p).astype(int)
    return y, p


def test_pct_names_the_fraction():
    assert [pct(f) for f in (0.01, 0.05, 0.1, 0.2, 0.025)] == ["1", "5", "10", "20", "2.5"]


def test_ranking_metrics_match_scikit_learn():
    y, p = scored()
    m = ranking_metrics(y, p, [0.05, 0.1])
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, p), abs=1e-12)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert m["base_rate"] == pytest.approx(y.mean())
    assert m["pr_auc_lift"] == pytest.approx(m["pr_auc"] / y.mean())
    assert m["rows"] == 2000 and m["positives"] == y.sum()


def test_top_k_counts_the_highest_scores():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    t = top_k(y, p, 0.3)  # rows 0, 1, 2: two of three positives
    assert t["k"] == 3
    assert t["precision"] == pytest.approx(2 / 3)
    assert t["recall"] == pytest.approx(2 / 3)
    assert t["lift"] == pytest.approx((2 / 3) / 0.3)


def test_top_k_flags_at_least_one_row():
    t = top_k([1, 0, 0], [0.2, 0.9, 0.1], 0.01)
    assert t["k"] == 1 and t["precision"] == 0.0


def test_top_k_breaks_ties_by_row_order():
    assert top_k([1, 0], [0.5, 0.5], 0.5)["precision"] == 1.0
    assert top_k([0, 1], [0.5, 0.5], 0.5)["precision"] == 0.0


def test_ranking_metrics_name_every_fraction():
    m = ranking_metrics(*scored(), [0.01, 0.2])
    assert {"precision_at_1", "recall_at_1", "lift_at_1", "precision_at_20", "lift_at_20"} <= set(m)


def test_a_perfect_ranking_scores_one():
    y = np.array([0, 0, 1, 1])
    m = ranking_metrics(y, [0.1, 0.2, 0.8, 0.9], [0.5])
    assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0 and m["precision_at_50"] == 1.0


def test_one_class_gives_nan_instead_of_failing():
    m = ranking_metrics([0, 0, 0], [0.1, 0.2, 0.3], [0.5])
    assert np.isnan(m["pr_auc"]) and np.isnan(m["roc_auc"]) and np.isnan(m["recall_at_50"])


@pytest.mark.parametrize(
    "y, p",
    [([0, 1], [0.1]), ([], []), ([0, 1], [0.1, np.nan]), ([[0, 1]], [[0.1, 0.2]])],
)
def test_bad_inputs_are_refused(y, p):
    with pytest.raises(ValueError):
        ranking_metrics(y, p, [0.1])


def test_a_fraction_outside_zero_one_is_refused():
    with pytest.raises(ValueError, match="frac"):
        top_k([0, 1], [0.1, 0.2], 0)


def test_brier_matches_scikit_learn_and_skill_is_against_the_base_rate():
    y, p = scored()
    m = probability_metrics(y, p, 10)
    assert m["brier"] == pytest.approx(brier_score_loss(y, p), abs=1e-12)
    base = y.mean()
    assert m["brier_reference"] == pytest.approx(brier_score_loss(y, np.full(len(y), base)))
    assert m["brier_skill"] == pytest.approx(1 - m["brier"] / m["brier_reference"])
    assert m["mean_predicted"] == pytest.approx(p.mean()) and m["observed_rate"] == base
    assert m["calibration_slope"] == calibration_line(y, p)["slope"]


def test_ece_by_hand_on_two_bins():
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    # bins: rows 0-2 (mean p 0.2, rate 1/3) and rows 3-5 (mean p 0.8, rate 1)
    assert ece(y, p, 2) == pytest.approx(0.5 * abs(1 / 3 - 0.2) + 0.5 * abs(1 - 0.8))


def test_reliability_table_bins_have_equal_mass_and_sorted_scores():
    y, p = scored(n=1003)
    table = reliability_table(y, p, 10)
    assert len(table) == 10
    assert {b["rows"] for b in table} == {100, 101}
    assert sum(b["rows"] for b in table) == 1003
    assert all(a["p_max"] <= b["p_min"] for a, b in zip(table, table[1:], strict=False))
    assert all(b["gap"] == pytest.approx(b["observed_rate"] - b["mean_predicted"]) for b in table)


def test_calibrated_scores_have_a_small_ece_and_shifted_ones_a_large_one():
    y, p = scored(n=50_000)
    assert ece(y, p, 10) < 0.01
    assert ece(y, np.clip(p + 0.1, 0, 1), 10) == pytest.approx(0.1, abs=0.01)


def test_more_bins_than_rows_uses_one_row_per_bin():
    assert len(reliability_table([0, 1, 1], [0.2, 0.5, 0.9], 10)) == 3


def test_probabilities_outside_zero_one_are_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        probability_metrics([0, 1], [0.5, 1.2], 10)


def test_calibrated_scores_have_slope_one_and_intercept_zero():
    y, p = scored(n=100_000, seed=1)
    line = calibration_line(y, p)
    assert line["slope"] == pytest.approx(1, abs=0.03)
    assert line["intercept"] == pytest.approx(0, abs=0.03)


def test_overconfident_scores_have_a_slope_below_one():
    y, p = scored(n=100_000, seed=2)
    logit = np.log(p / (1 - p))
    too_sure = 1 / (1 + np.exp(-2 * logit))  # the true log-odds, doubled
    assert calibration_line(y, too_sure)["slope"] == pytest.approx(0.5, abs=0.03)


def test_calibration_line_matches_an_unpenalised_logistic_fit():
    y, p = scored(n=5000, seed=3)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(C=1e12, max_iter=10_000, tol=1e-12).fit(logit, y)
    line = calibration_line(y, p)
    assert line["slope"] == pytest.approx(fit.coef_[0, 0], abs=1e-4)
    assert line["intercept"] == pytest.approx(fit.intercept_[0], abs=1e-4)


def test_calibration_line_of_one_class_is_nan():
    assert np.isnan(calibration_line([1, 1], [0.2, 0.4])["slope"])
