"""Leakage test 6 (docs/ML_PIPELINE_SPEC.md §17.1, §4.2): reordered or unseen categorical levels give the same
predictions for seen levels. A row's prediction depends on its own values only, never on which other rows (or
levels) share the batch or their order. (Leakage test 4's member half, OOF predictions from fits without the
facility, is covered by the train tests.)"""

import numpy as np
import pandas as pd
import pytest

from precursorintelligence.common.config import FeatureSpec, LrParams
from precursorintelligence.training.contracts import PlattCalibration
from precursorintelligence.training.data import MISSING_LEVEL
from precursorintelligence.training.ensemble import PrecursorModel, build_meta, candidate_weights
from precursorintelligence.training.members import LgbmMember, LrMember, categorical_levels

LGBM = {
    "n_estimators": 30,
    "num_leaves": 7,
    "min_child_samples": 5,
    "seed": 1,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 1,
    "learning_rate": 0.1,
}


def spec(name, dtype):
    return FeatureSpec(
        name=name,
        family="x",
        source_table="x",
        formula="x",
        first_available_vintage="2022-01",
        dtype=dtype,
        model_use="model",
    )


# two categoricals, so the test also covers more than one encoder
FEATURES = [spec("a", "f64"), spec("b", "i64"), spec("d", "str"), spec("e", "str")]


def model_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    a[rng.random(n) < 0.2] = np.nan
    b = rng.integers(0, 10, size=n).astype("float64")
    d = rng.choice(["p", "q", "r", MISSING_LEVEL], size=n, p=[0.45, 0.3, 0.15, 0.1]).astype(object)
    e = rng.choice(["k1", "k2", "k3"], size=n).astype(object)
    z = (
        np.nan_to_num(a)
        + 0.2 * b
        - 1.0
        + np.where(d == "p", 0.8, 0.0)
        + np.where(e == "k2", -0.7, 0)
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    x = pd.DataFrame(
        {"a": a, "b": b, "d": pd.Series(d, dtype=object), "e": pd.Series(e, dtype=object)}
    )
    return x, y


@pytest.fixture(scope="module")
def fitted():
    x, y = model_frame()
    levels = categorical_levels(x, FEATURES)
    lgbm = LgbmMember.fit(x, y, FEATURES, LGBM, levels=levels)
    lr = LrMember.fit(x, y, FEATURES, LrParams(C=0.3, max_iter=500, onehot_min_frequency=5))
    weights = candidate_weights("ensemble", {"lgbm": 0.5, "lr": 0.5})
    meta = build_meta(
        "ensemble", FEATURES, levels, weights, {"eval": ["2023-01"]}, recipe_hash="test"
    )
    platt = PlattCalibration(a=0.9, b=0.1, identity=False, fit_rows=len(y), fit_months=["2023-01"])
    model = PrecursorModel(meta, platt, lgbm=lgbm, lr=lr)
    # the paths fold fits and stored packages take: levels from the fit's own rows, and a package reloaded from
    # its files (levels from meta.json plus LightGBM's own pandas_categorical)
    own_levels = LgbmMember.fit(x, y, FEATURES, LGBM)
    reloaded = PrecursorModel.from_files(model.to_files())
    return x, {
        "lgbm": lgbm.logit,
        "lr": lr.logit,
        "lgbm_own_levels": own_levels.logit,
        "model_p": lambda f: model.predict_frame(f)["p"].to_numpy(),
        "model_contrib": lambda f: model.contributions(f)[1].to_numpy(),
        "reloaded_p": lambda f: reloaded.predict_frame(f)["p"].to_numpy(),
        "reloaded_contrib": lambda f: reloaded.contributions(f)[1].to_numpy(),
    }


def same(actual, expected):
    """Equal up to floating-point summation order (BLAS may block a matrix product by batch size)."""
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def shuffled_with_unseen(x, seed=1):
    """The rows in a new order, interleaved with copies carrying levels no fit saw; returns (frame, where each
    original row now is)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    unseen = x.sample(n=n // 2, random_state=seed).reset_index(drop=True)
    unseen["d"] = pd.Series(
        rng.choice(["zz", "aa", "never"], size=len(unseen)), dtype=object, index=unseen.index
    )
    unseen["e"] = pd.Series(["k0"] * len(unseen), dtype=object, index=unseen.index)
    batch = pd.concat([x.reset_index(drop=True), unseen], ignore_index=True)
    order = rng.permutation(len(batch))
    mixed = batch.iloc[order].reset_index(drop=True)
    position = np.empty(len(batch), dtype=int)
    position[order] = np.arange(len(batch))
    return mixed, position[:n]


@pytest.mark.parametrize(
    "which",
    ["lgbm", "lr", "lgbm_own_levels", "model_p", "model_contrib", "reloaded_p", "reloaded_contrib"],
)
def test_seen_rows_score_the_same_in_any_batch(fitted, which):
    x, fns = fitted
    fn = fns[which]
    alone = fn(x)
    mixed, where = shuffled_with_unseen(x)
    out = fn(mixed)
    assert np.isfinite(out).all()
    same(out[where], alone)


@pytest.mark.parametrize("which", ["lgbm", "lr", "model_p"])
def test_one_row_at_a_time_matches_the_batch(fitted, which):
    x, fns = fitted
    fn = fns[which]
    alone = fn(x)
    for i in (0, 7, 42, 299):
        same(fn(x.iloc[[i]]), alone[[i]])


@pytest.mark.parametrize("which", ["lgbm", "lr", "model_p"])
def test_batch_with_only_some_levels_matches(fitted, which):
    """A batch where some seen levels are absent (so a pandas Categorical built from the batch would get other
    codes) still scores each row the same."""
    x, fns = fitted
    fn = fns[which]
    alone = fn(x)
    keep = (x["d"] == "r").to_numpy() & (x["e"] == "k3").to_numpy()
    same(fn(x[keep]), alone[keep])
    # and a pandas Categorical input with its levels in another order gives the same scores
    cat = x.copy()
    cat["d"] = pd.Categorical(cat["d"], categories=["r", MISSING_LEVEL, "q", "p"])
    same(fn(cat), alone)
