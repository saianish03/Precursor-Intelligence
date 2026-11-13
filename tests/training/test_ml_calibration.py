"""training/calibration.py: Platt fit, the degenerate fallback, ECE and diagnostics (docs/ML_PIPELINE_SPEC.md §5,
§17.1: "Platt lowers ECE on a miscalibrated fixture")."""

import numpy as np
import pytest

from precursorintelligence.training.calibration import (
    apply_platt,
    brier,
    diagnostics,
    ece,
    fit_platt,
    recent_months,
    sigmoid,
)

MONTHS = ["2023-01", "2023-02"]


def miscalibrated(n=20000, seed=0):
    """True p = sigmoid(z); the scores are 3z + 1 (over-confident and shifted)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.5, size=n)
    y = (rng.random(n) < sigmoid(z)).astype(int)
    return z, 3.0 * z + 1.0, y


def test_platt_lowers_ece_and_brier_on_a_miscalibrated_fixture():
    _, scores, y = miscalibrated()
    cal = fit_platt(scores, y, MONTHS)
    assert not cal.identity
    assert cal.fit_rows == len(y)
    assert cal.fit_months == MONTHS
    assert cal.fitted_a is None and cal.fitted_b is None
    p_raw = sigmoid(scores)
    p_cal = sigmoid(apply_platt(cal, scores))
    assert ece(y, p_cal) < 0.5 * ece(y, p_raw)
    assert ece(y, p_cal) < 0.02
    assert brier(y, p_cal) < brier(y, p_raw)


def test_platt_recovers_the_true_map():
    _, scores, y = miscalibrated()
    cal = fit_platt(scores, y, MONTHS)
    # sigmoid(a(3z + 1) + b) = sigmoid(z) needs a = 1/3, b = -1/3
    assert cal.a == pytest.approx(1 / 3, abs=0.03)
    assert cal.b == pytest.approx(-1 / 3, abs=0.05)


def test_apply_platt_is_affine():
    from precursorintelligence.training.contracts import PlattCalibration

    cal = PlattCalibration(a=2.0, b=-0.5, identity=False, fit_rows=1, fit_months=[])
    np.testing.assert_allclose(apply_platt(cal, np.array([0.0, 1.0, -2.0])), [-0.5, 1.5, -4.5])


def test_anticorrelated_scores_fall_back_to_identity():
    z, _, y = miscalibrated(n=5000)
    cal = fit_platt(-3.0 * z, y, MONTHS)
    assert cal.identity
    assert (cal.a, cal.b) == (1.0, 0.0)
    assert cal.fitted_a is not None and cal.fitted_a <= 0
    assert cal.fitted_b is not None
    assert cal.fit_rows == 5000
    assert cal.fit_months == MONTHS


@pytest.mark.parametrize("y", [np.zeros(50), np.ones(50)])
def test_single_class_labels_give_identity(y):
    cal = fit_platt(np.linspace(-2, 2, 50), y, MONTHS)
    assert cal.identity
    assert (cal.a, cal.b) == (1.0, 0.0)
    assert cal.fitted_a is None
    assert cal.fit_rows == 50


def test_no_rows_give_identity():
    cal = fit_platt(np.array([]), np.array([]), [])
    assert cal.identity and cal.fit_rows == 0


def test_ece_hand_computed():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.9, 0.2, 0.7, 0.1])
    # sorted by p: (0.1, 1), (0.2, 1) | (0.7, 0), (0.9, 0)
    # bin 1: |1 - 0.15| = 0.85; bin 2: |0 - 0.8| = 0.8; each weighted 1/2
    assert ece(y, p, bins=2) == pytest.approx(0.825)
    # one row per bin (bins > rows clamps to n): mean |y - p|
    assert ece(y, p, bins=10) == pytest.approx(np.mean(np.abs(y - p)))
    assert ece(y, np.where(y == 1, 1.0, 0.0), bins=2) == pytest.approx(0.0)
    assert np.isnan(ece(np.array([]), np.array([])))


def test_brier_hand_computed():
    assert brier(np.array([0, 1]), np.array([0.2, 0.6])) == pytest.approx((0.04 + 0.16) / 2)


def test_recent_months_picks_the_newest_distinct_months():
    months = ["2023-03", "2023-01", "2023-03", "2023-02", "2022-12", "2023-01"]
    assert recent_months(months, 2) == ["2023-02", "2023-03"]
    assert recent_months(months, 10) == ["2022-12", "2023-01", "2023-02", "2023-03"]
    assert recent_months([], 3) == []


def test_diagnostics_numbers():
    _, scores, y = miscalibrated(n=5000)
    cal = fit_platt(scores, y, MONTHS)
    d = diagnostics("lgbm", "fit_months", scores, y, cal)
    assert d.candidate == "lgbm" and d.rows == "fit_months"
    for v in (d.brier_raw, d.brier_platt, d.brier_isotonic, d.ece_raw, d.ece_platt, d.ece_isotonic):
        assert 0.0 <= v <= 1.0
    assert d.brier_raw == pytest.approx(brier(y, sigmoid(scores)))
    assert d.ece_platt == pytest.approx(ece(y, sigmoid(apply_platt(cal, scores))))
    assert d.ece_platt < d.ece_raw
    assert d.brier_platt < d.brier_raw
    # isotonic, on its own fit rows, is the best monotone map: never worse than raw (or Platt)
    assert d.brier_isotonic <= d.brier_raw
    assert d.brier_isotonic <= d.brier_platt + 1e-12


def test_diagnostics_isotonic_fit_elsewhere_and_single_class():
    _, scores, y = miscalibrated(n=2000)
    cal = fit_platt(scores, y, MONTHS)
    d = diagnostics("lr", "all_oof", scores, y, cal, isotonic_fit=(scores[:1000], y[:1000]))
    assert 0.0 <= d.brier_isotonic <= 1.0
    # a fit set with one class: isotonic falls back to the raw probabilities
    d1 = diagnostics("lr", "all_oof", scores, y, cal, isotonic_fit=(scores[:5], np.zeros(5)))
    assert d1.brier_isotonic == pytest.approx(d1.brier_raw)
    assert d1.ece_isotonic == pytest.approx(d1.ece_raw)
