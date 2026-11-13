"""training/members.py: LR preprocessing, exact contributions, serialisation, LightGBM determinism
(docs/ML_PIPELINE_SPEC.md §4.2, §8.1, §17.1 unit and model suites)."""

import numpy as np
import pandas as pd
import pytest

from precursorintelligence.common.config import FeatureSpec, LrParams
from precursorintelligence.training.data import MISSING_LEVEL
from precursorintelligence.training.members import (
    LgbmMember,
    LrMember,
    categorical_levels,
)

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


FEATURES = [spec("a", "f64"), spec("b", "i64"), spec("c", "bool"), spec("d", "str")]


def model_frame(n=300, seed=0):
    """A ``to_model_frame``-shaped frame: float64 numerics (``a`` has nulls), object ``d`` with a rare level
    and nulls as MISSING_LEVEL; and a label that depends on all of them."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    a[rng.random(n) < 0.2] = np.nan
    b = rng.integers(0, 10, size=n).astype("float64")
    c = (rng.random(n) < 0.4).astype("float64")
    d = rng.choice(["p", "q", "r", MISSING_LEVEL], size=n, p=[0.45, 0.3, 0.15, 0.1]).astype(object)
    d[:3] = "rare"  # 3 rows: below onehot_min_frequency=5
    z = np.nan_to_num(a) + 0.2 * b - 1.0 + c + np.where(d == "p", 0.8, 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    x = pd.DataFrame({"a": a, "b": b, "c": c, "d": pd.Series(d, dtype=object)})
    return x, y


@pytest.fixture(scope="module")
def data():
    return model_frame()


@pytest.fixture(scope="module")
def lr(data):
    x, y = data
    return LrMember.fit(x, y, FEATURES, LrParams(C=0.3, max_iter=500, onehot_min_frequency=5))


@pytest.fixture(scope="module")
def lgbm(data):
    x, y = data
    return LgbmMember.fit(x, y, FEATURES, LGBM, levels=categorical_levels(x, FEATURES))


# ------------------------------------------------------------------ LR preprocessing


def test_lr_numeric_with_nulls_gets_a_missing_indicator(lr):
    src = lr.design_sources
    # a (has nulls) appears twice: scaled value + missing indicator; b and c (no nulls) once each
    assert src.count("a") == 2
    assert src.count("b") == 1
    assert src.count("c") == 1
    assert len(src) == lr.design_means.size == lr.coef.size


def test_lr_median_imputation(data, lr):
    x, _ = data
    imputer = lr.pipeline.named_steps["pre"].named_transformers_["num"].steps[0][1]
    assert imputer.statistics_[0] == pytest.approx(float(np.nanmedian(x["a"])))
    one = x.iloc[[0]].copy()
    one["a"] = np.nan
    other = one.copy()
    other["a"] = float(np.nanmedian(x["a"]))
    za, zb = lr.design(one), lr.design(other)
    ia = lr.design_sources.index("a")  # the value column comes first, the indicator after
    assert za[0, ia] == pytest.approx(zb[0, ia])
    indicator = [i for i, s in enumerate(lr.design_sources) if s == "a"][1]
    assert za[0, indicator] != zb[0, indicator]


def test_lr_all_nan_training_column_does_not_crash(data):
    x, y = data
    x = x.copy()
    x["b"] = np.nan
    m = LrMember.fit(x, y, FEATURES, LrParams(C=0.3, max_iter=500, onehot_min_frequency=5))
    assert "b" in m.design_sources
    out = m.logit(x.assign(b=3.0))  # values at predict for a column empty in training
    assert np.isfinite(out).all()
    base, contrib = m.contributions(x)
    np.testing.assert_allclose(base + contrib.sum(axis=1).to_numpy(), m.logit(x), atol=1e-8)


def test_lr_numerics_are_standardised(data, lr):
    x, _ = data
    z = lr.design(x)
    for name in ("a", "b", "c"):
        j = lr.design_sources.index(name)  # the scaled value column
        assert abs(z[:, j].mean()) < 1e-9
        assert z[:, j].std() == pytest.approx(1.0, abs=1e-9)


def test_lr_rare_levels_are_pooled_as_infrequent(lr):
    enc = lr.pipeline.named_steps["pre"].named_transformers_["cat0"]
    assert list(enc.infrequent_categories_[0]) == ["rare"]
    names = list(enc.get_feature_names_out())
    assert any("infrequent" in n for n in names)
    assert lr.design_sources.count("d") == len(names)


def test_lr_null_categorical_is_the_missing_level(data, lr):
    x, _ = data
    enc = lr.pipeline.named_steps["pre"].named_transformers_["cat0"]
    assert MISSING_LEVEL in list(enc.categories_[0])
    raw = x.iloc[[0]].copy()
    raw["d"] = pd.Series([None], dtype=object, index=raw.index)
    filled = raw.copy()
    filled["d"] = pd.Series([MISSING_LEVEL], dtype=object, index=raw.index)
    np.testing.assert_array_equal(lr.design(raw), lr.design(filled))


def test_lr_unseen_level_at_predict(data, lr):
    x, _ = data
    unseen = x.iloc[:5].copy()
    unseen["d"] = pd.Series(["never-seen"] * 5, dtype=object, index=unseen.index)
    out = lr.logit(unseen)
    assert np.isfinite(out).all()
    # handled like the pooled infrequent level
    rare = unseen.copy()
    rare["d"] = pd.Series(["rare"] * 5, dtype=object, index=unseen.index)
    np.testing.assert_allclose(out, lr.logit(rare), atol=1e-12)


# ------------------------------------------------------------------ contributions


def test_lr_contributions_are_exact(data, lr):
    x, _ = data
    base, contrib = lr.contributions(x)
    assert list(contrib.columns) == ["a", "b", "c", "d"]
    np.testing.assert_allclose(base + contrib.sum(axis=1).to_numpy(), lr.logit(x), atol=1e-8)
    # over the training rows each contribution averages to zero (centred on the design means)
    np.testing.assert_allclose(contrib.mean(axis=0).to_numpy(), 0.0, atol=1e-10)


def test_lgbm_contributions_are_exact(data, lgbm):
    x, _ = data
    base, contrib = lgbm.contributions(x)
    assert list(contrib.columns) == ["a", "b", "c", "d"]
    np.testing.assert_allclose(base + contrib.sum(axis=1).to_numpy(), lgbm.logit(x), atol=1e-6)


# ------------------------------------------------------------------ serialisation and determinism


def test_lr_round_trip(data, lr):
    x, _ = data
    back = LrMember.from_bytes(lr.to_bytes())
    np.testing.assert_array_equal(back.logit(x), lr.logit(x))
    assert back.design_sources == lr.design_sources
    assert [f.name for f in back.features] == ["a", "b", "c", "d"]


def test_lr_refuses_other_format():
    import io

    import joblib

    buf = io.BytesIO()
    joblib.dump({"format": 99}, buf)
    with pytest.raises(ValueError, match="format"):
        LrMember.from_bytes(buf.getvalue())


def test_lgbm_round_trip(data, lgbm):
    x, _ = data
    back = LgbmMember.from_bytes(lgbm.to_bytes(), FEATURES, lgbm.levels)
    np.testing.assert_array_equal(back.logit(x), lgbm.logit(x))


def test_lgbm_fit_is_deterministic(data, lgbm):
    x, y = data
    again = LgbmMember.fit(x, y, FEATURES, LGBM, levels=categorical_levels(x, FEATURES))
    assert again.to_bytes() == lgbm.to_bytes()


def test_categorical_levels_are_sorted_and_include_missing(data):
    x, _ = data
    assert categorical_levels(x, FEATURES) == {"d": sorted(["p", "q", "r", "rare", MISSING_LEVEL])}


def test_an_unknown_lgbm_key_raises_instead_of_training(data):
    """LightGBM ignores an unknown parameter (and verbose -1 hides its warning): the member refuses it."""
    x, y = data
    with pytest.raises(ValueError, match="num_leave"):
        LgbmMember.fit(x, y, FEATURES, {**LGBM, "num_leave": 3})
    with pytest.raises(ValueError, match="n_estimators"):
        LgbmMember.fit(x, y, FEATURES, {k: v for k, v in LGBM.items() if k != "n_estimators"})
