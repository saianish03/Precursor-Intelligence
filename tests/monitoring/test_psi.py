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
