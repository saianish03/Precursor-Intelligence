"""training/ensemble.py: the log-odds ensemble, PrecursorModel scoring, contributions after Platt, and the package
round trips (docs/ML_PIPELINE_SPEC.md §4.2, §4.5, §8.1, §17.1 model suite)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from precursorintelligence.common.config import FeatureSpec, LrParams
from precursorintelligence.common.io.object_store import LocalObjectStore
from precursorintelligence.training.calibration import sigmoid
from precursorintelligence.training.contracts import (
    PACKAGE_CALIBRATION,
    PACKAGE_LGBM,
    PACKAGE_LR,
    PACKAGE_META,
    PlattCalibration,
)
from precursorintelligence.training.data import MISSING_LEVEL
from precursorintelligence.training.ensemble import (
    PackageError,
    PrecursorModel,
    build_meta,
    candidate_weights,
    combine,
    read_package,
    write_package,
)
from precursorintelligence.training.members import LgbmMember, LrMember, categorical_levels

PACKAGE_V1 = Path(__file__).parent / "data" / "package_v1"
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
ENSEMBLE_WEIGHTS = {"lgbm": 3.0, "lr": 1.0}  # deliberately not normalised


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
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    a[rng.random(n) < 0.2] = np.nan
    b = rng.integers(0, 10, size=n).astype("float64")
    c = (rng.random(n) < 0.4).astype("float64")
    d = rng.choice(["p", "q", "r", MISSING_LEVEL], size=n, p=[0.45, 0.3, 0.15, 0.1]).astype(object)
    z = np.nan_to_num(a) + 0.2 * b - 1.0 + c + np.where(d == "p", 0.8, 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": pd.Series(d, dtype=object)}), y


PLATT = PlattCalibration(a=0.7, b=-0.3, identity=False, fit_rows=300, fit_months=["2023-01"])


@pytest.fixture(scope="module")
def data():
    return model_frame()


@pytest.fixture(scope="module")
def members(data):
    x, y = data
    levels = categorical_levels(x, FEATURES)
    lgbm = LgbmMember.fit(x, y, FEATURES, LGBM, levels=levels)
    lr = LrMember.fit(x, y, FEATURES, LrParams(C=0.3, max_iter=500, onehot_min_frequency=5))
    return levels, lgbm, lr


def package(members, candidate):
    levels, lgbm, lr = members
    weights = candidate_weights(candidate, ENSEMBLE_WEIGHTS)
    meta = build_meta(
        candidate, FEATURES, levels, weights, {"eval": ["2023-01"]}, recipe_hash="test"
    )
    return PrecursorModel(
        meta,
        PLATT,
        lgbm=lgbm if "lgbm" in weights else None,
        lr=lr if "lr" in weights else None,
    )


@pytest.fixture(scope="module")
def ensemble(members):
    return package(members, "ensemble")


# ------------------------------------------------------------------ the log-odds ensemble


def test_candidate_weights_normalise():
    assert candidate_weights("ensemble", {"lgbm": 3.0, "lr": 1.0}) == {"lgbm": 0.75, "lr": 0.25}
    assert candidate_weights("lgbm", {"lgbm": 3.0, "lr": 1.0}) == {"lgbm": 1.0}
    assert candidate_weights("lr", {"lgbm": 3.0, "lr": 1.0}) == {"lr": 1.0}
    w = candidate_weights("ensemble", {"lgbm": 0.2, "lr": 0.3})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["lr"] == pytest.approx(0.6)


def test_combine_is_the_weighted_mean_of_member_logits():
    lg, lr = np.array([1.0, -2.0, 0.5]), np.array([3.0, 0.0, -1.5])
    np.testing.assert_allclose(
        combine({"lgbm": lg, "lr": lr}, {"lgbm": 3.0, "lr": 1.0}), (3 * lg + lr) / 4
    )
    np.testing.assert_allclose(combine({"lgbm": lg, "lr": lr}, {"lgbm": 1.0}), lg)


def test_predict_frame_columns_and_calibration(data, members, ensemble):
    x, _ = data
    _, lgbm, lr = members
    out = ensemble.predict_frame(x)
    assert list(out.columns) == ["p", "logit", "logit_raw", "logit_lgbm", "logit_lr"]
    np.testing.assert_array_equal(out["logit_lgbm"], lgbm.logit(x))
    np.testing.assert_array_equal(out["logit_lr"], lr.logit(x))
    np.testing.assert_allclose(out["logit_raw"], 0.75 * lgbm.logit(x) + 0.25 * lr.logit(x))
    np.testing.assert_allclose(out["logit"], PLATT.a * out["logit_raw"] + PLATT.b)
    np.testing.assert_allclose(out["p"], sigmoid(PLATT.a * out["logit_raw"].to_numpy() + PLATT.b))
    assert out.index.equals(x.index)


def test_single_member_packages_have_one_member_column(data, members):
    x, _ = data
    for candidate in ("lgbm", "lr"):
        out = package(members, candidate).predict_frame(x)
        assert list(out.columns) == ["p", "logit", "logit_raw", f"logit_{candidate}"]
        np.testing.assert_array_equal(out["logit_raw"], out[f"logit_{candidate}"])


def test_predict_pyfunc_entry_point(data, ensemble):
    x, _ = data
    pd.testing.assert_frame_equal(ensemble.predict(None, x), ensemble.predict_frame(x))


def test_predict_index_is_kept(data, ensemble):
    x, _ = data
    sub = x.iloc[10:20]
    out = ensemble.predict_frame(sub)
    assert out.index.equals(sub.index)
    np.testing.assert_allclose(out["p"], ensemble.predict_frame(x)["p"].iloc[10:20])


# ------------------------------------------------------------------ contributions after Platt


@pytest.mark.parametrize("candidate", ["ensemble", "lgbm", "lr"])
def test_contributions_add_up_to_the_calibrated_logit(data, members, candidate):
    x, _ = data
    model = package(members, candidate)
    for fn in (model.contributions, model.explain):
        base, contrib = fn(x)
        assert list(contrib.columns) == ["a", "b", "c", "d"]
        np.testing.assert_allclose(
            base + contrib.sum(axis=1).to_numpy(),
            model.predict_frame(x)["logit"].to_numpy(),
            atol=1e-6,
        )


def test_contributions_are_platt_scaled_member_contributions(data, members):
    x, _ = data
    _, _, lr = members
    model = package(members, "lr")
    base, contrib = model.contributions(x)
    lb, lc = lr.contributions(x)
    np.testing.assert_allclose(base, PLATT.a * lb + PLATT.b)
    np.testing.assert_allclose(contrib.to_numpy(), PLATT.a * lc.to_numpy())


# ------------------------------------------------------------------ persistence


@pytest.mark.parametrize("candidate", ["ensemble", "lgbm", "lr"])
def test_to_files_round_trip(data, members, candidate):
    x, _ = data
    model = package(members, candidate)
    files = model.to_files()
    expected = {PACKAGE_META, PACKAGE_CALIBRATION}
    expected |= {PACKAGE_LGBM} if candidate != "lr" else set()
    expected |= {PACKAGE_LR} if candidate != "lgbm" else set()
    assert set(files) == expected
    back = PrecursorModel.from_files(files)
    pd.testing.assert_frame_equal(back.predict_frame(x), model.predict_frame(x))
    assert back.meta == model.meta
    assert back.platt == model.platt


def test_write_dir_read_dir_round_trip(tmp_path, data, ensemble):
    x, _ = data
    paths = ensemble.write_dir(tmp_path / "pkg")
    assert set(paths) == {PACKAGE_META, PACKAGE_CALIBRATION, PACKAGE_LGBM, PACKAGE_LR}
    back = PrecursorModel.read_dir(tmp_path / "pkg")
    pd.testing.assert_frame_equal(back.predict_frame(x), ensemble.predict_frame(x))


def test_write_package_read_package_round_trip(tmp_path, data, ensemble):
    x, _ = data
    store = LocalObjectStore(tmp_path, immutable=None)
    prefix = "ml/runs/r1/models/eval/ensemble/"
    shas = write_package(store, prefix, ensemble)
    assert set(shas) == {PACKAGE_META, PACKAGE_CALIBRATION, PACKAGE_LGBM, PACKAGE_LR}
    assert all(len(s) == 64 for s in shas.values())
    back = read_package(store, prefix)
    pd.testing.assert_frame_equal(back.predict_frame(x), ensemble.predict_frame(x))
    # writing again (a retried task) overwrites rather than failing
    assert write_package(store, prefix, ensemble) == shas


def test_load_context_rebuilds_the_model(tmp_path, data, ensemble):
    x, _ = data
    paths = ensemble.write_dir(tmp_path / "artifacts")

    class Context:
        artifacts = paths

    empty = PrecursorModel()
    empty.load_context(Context())
    pd.testing.assert_frame_equal(empty.predict(Context(), x), ensemble.predict_frame(x))


def test_newer_package_format_is_refused(ensemble):
    files = ensemble.to_files()
    meta = json.loads(files[PACKAGE_META])
    meta["package_format"] = 2
    files[PACKAGE_META] = json.dumps(meta).encode("utf-8")
    with pytest.raises(PackageError, match="package_format 2"):
        PrecursorModel.from_files(files)


@pytest.mark.parametrize("missing", [PACKAGE_LGBM, PACKAGE_LR, PACKAGE_META, PACKAGE_CALIBRATION])
def test_missing_file_is_refused(ensemble, missing):
    files = ensemble.to_files()
    del files[missing]
    with pytest.raises(PackageError):
        PrecursorModel.from_files(files)


def test_missing_member_object_is_refused(members):
    levels, lgbm, _ = members
    meta = build_meta(
        "ensemble", FEATURES, levels, {"lgbm": 0.5, "lr": 0.5}, {"eval": ["2023-01"]}, "test"
    )
    with pytest.raises(PackageError, match="lr"):
        PrecursorModel(meta, PLATT, lgbm=lgbm)
    with pytest.raises(PackageError, match="calibration"):
        PrecursorModel(meta, None, lgbm=lgbm, lr=members[2])


def test_read_package_missing_member_is_refused(tmp_path, ensemble):
    store = LocalObjectStore(tmp_path, immutable=None)
    files = ensemble.to_files()
    del files[PACKAGE_LR]
    for name, data in files.items():
        store.put_bytes(f"p/{name}", data)
    with pytest.raises(PackageError, match=PACKAGE_LR):
        read_package(store, "p/")


def test_stored_version_1_package_still_loads():
    """A package written by an earlier version (tests/training/data/make_package_v1.py) keeps loading and giving
    the same probabilities. Never regenerate it to make this test pass."""
    expected = json.loads((PACKAGE_V1 / "expected.json").read_text(encoding="utf-8"))
    model = PrecursorModel.read_dir(PACKAGE_V1)
    assert model.meta is not None and model.meta.package_format == 1
    rows = pd.DataFrame(expected["rows"])
    x = pd.DataFrame(
        {
            "a": pd.to_numeric(rows["a"]).astype("float64"),
            "b": rows["b"].astype("float64"),
            "c": rows["c"].astype("float64"),
            "d": pd.Series(rows["d"].tolist(), dtype=object),
        }
    )
    np.testing.assert_allclose(model.predict_frame(x)["p"].to_numpy(), expected["p"], atol=1e-9)
