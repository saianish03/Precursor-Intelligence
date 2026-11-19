"""monitoring/drift.psi (§12.1): quantile bins of the reference, levels for categoricals, nulls as their own bin."""

import numpy as np
import polars as pl
import pytest

from precursorintelligence.monitoring.drift import psi

rng = np.random.default_rng(0)


def test_the_same_distribution_has_psi_near_zero():
    a = pl.Series(rng.normal(size=5000))
    b = pl.Series(rng.normal(size=5000))
    assert psi(a, b) < 0.01


def test_a_shifted_distribution_crosses_the_drift_limit():
    a = pl.Series(rng.normal(size=5000))
    assert psi(a, a + 1.0) > 0.25


def test_nulls_are_a_bin_of_their_own():
    ref = pl.Series([1.0, 2.0, 3.0, 4.0] * 250)
    cur = pl.Series([1.0, 2.0, 3.0, 4.0] * 125 + [None] * 500)
    assert psi(ref, cur) > 0.25  # half the rows moved to the null bin


def test_categoricals_compare_levels():
    ref = pl.Series(["A"] * 500 + ["B"] * 500)
    assert psi(ref, ref) == pytest.approx(0.0)
    assert psi(ref, pl.Series(["A"] * 900 + ["B"] * 100)) > 0.25
    assert psi(ref, pl.Series(["A"] * 500 + ["C"] * 500)) > 0.25  # a new level counts


def test_a_constant_column_is_one_bin():
    ones = pl.Series([1] * 100)
    assert psi(ones, ones) == pytest.approx(0.0)


def test_an_empty_month_is_refused():
    with pytest.raises(ValueError, match="rows in both"):
        psi(pl.Series([], dtype=pl.Float64), pl.Series([1.0]))


def flag_rate(shift_sd, draws=100):
    """The share of months in which a state-level feature (one value per state, as state_gplus_rate_12m) crosses
    the drift limit. 51 states shaped like the real ones (rates bunched around 0.18 with a long right tail, the
    largest state 7% of homes); each month every state's rate moves at random with sd 0.13 x the spread across
    states (as measured between 2026-09 and 2026-10), plus ``shift_sd`` x that spread."""
    g = np.random.default_rng(0)
    sizes = np.round(g.lognormal(5.3, 0.8, 51)).astype(int)
    rates = g.gamma(3.0, 0.065, 51)
    spread = rates.std()
    ref = pl.Series(np.repeat(rates, sizes))
    moved = [rates + g.normal(shift_sd * spread, 0.13 * spread, 51) for _ in range(draws)]
    return np.mean([psi(ref, pl.Series(np.repeat(m, sizes))) > 0.25 for m in moved])


def test_a_state_level_feature_rarely_drifts_in_a_normal_month():
    # #73 B2: quantile edges sat on big states' values, so a state moving a little emptied a bin it held almost
    # alone (PSI 0.96 on the real 2026-10 month). Quantile bins flag 67% of these months, equal-width bins 15%
    # (1-4% for the real state features)
    assert flag_rate(0.0) < 0.3


def test_a_state_level_feature_that_really_moved_still_drifts():
    assert flag_rate(0.5) == 1.0


def test_integer_counts_keep_their_quantile_bins():
    # counts repeat the same values every month: a moved share must still show
    ref = pl.Series([0] * 600 + [1] * 250 + [2] * 100 + [3] * 50)
    assert psi(ref, ref) == pytest.approx(0.0)
    assert psi(ref, pl.Series([0] * 300 + [1] * 300 + [2] * 250 + [3] * 150)) > 0.25


@pytest.mark.parametrize(("ref", "cur"), [(17.5, 18.5), (17.5, 16.5), (1, 2)])
def test_a_one_value_column_that_moved_as_a_whole_drifts(ref, cur):
    # one value for every home (qm_age_months_* is one per month): the whole column moving is real drift
    assert psi(pl.Series([ref] * 1000), pl.Series([cur] * 1000)) > 0.25
