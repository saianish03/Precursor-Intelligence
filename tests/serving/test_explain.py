"""serving/explain.py: exact contributions, reason codes and labels (docs/ML_PIPELINE_SPEC.md §8, §17.1)."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

from precursorintelligence.common.config import (
    FeatureSpec,
    LgbmParams,
    LrParams,
    load_feature_registry,
    project_root,
)
from precursorintelligence.serving import explain
from precursorintelligence.serving.explain import ReasonLabel
from precursorintelligence.training.contracts import PlattCalibration
from precursorintelligence.training.ensemble import PrecursorModel, build_meta, candidate_weights
from precursorintelligence.training.members import LgbmMember, LrMember, categorical_levels

TOL = 1e-6


def _spec(name, dtype):
    return FeatureSpec(
        name=name,
        family="fam_" + name[0],
        source_table="",
        formula="",
        first_available_vintage="2021-01",
        dtype=dtype,
        model_use="model",
    )


FEATURES = [
    _spec("n_cit", "f64"),
    _spec("hprd", "f64"),
    _spec("overdue", "bool"),
    _spec("state", "str"),
]


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "n_cit": rng.poisson(3, n).astype(float),
            "hprd": rng.normal(3.5, 0.6, n),
            "overdue": rng.integers(0, 2, n).astype(float),
            "state": rng.choice(["CA", "NY", "TX", "__missing__"], n).astype(object),
        }
    )
    x.loc[rng.random(n) < 0.15, "hprd"] = np.nan  # nulls exercise LR's missing indicator
    logit = 0.4 * x["n_cit"] - 0.5 * x["hprd"].fillna(3.5) + 0.8 * x["overdue"] - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int).to_numpy()
    return x, y


@pytest.fixture(scope="module")
def members():
    x, y = _frame(1500, 0)
    lgbm = LgbmMember.fit(
        x,
        y,
        FEATURES,
        LgbmParams(n_estimators=40, num_leaves=8, min_child_samples=20, seed=1, num_threads=1),
    )
    lr = LrMember.fit(x, y, FEATURES, LrParams(C=0.3, max_iter=500, onehot_min_frequency=5))
    return lgbm, lr, categorical_levels(x, FEATURES)


def _model(members, candidate, a=0.8, b=-0.2):
    lgbm, lr, levels = members
    weights = candidate_weights(candidate, {"lgbm": 0.5, "lr": 0.5})
    meta = build_meta(candidate, FEATURES, levels, weights, {"eval": [], "refit": []}, "test")
    platt = PlattCalibration(a=a, b=b, identity=False, fit_rows=1, fit_months=[])
    return PrecursorModel(meta, platt, lgbm=lgbm, lr=lr)


@pytest.mark.parametrize("candidate", ["lgbm", "lr", "ensemble"])
def test_contributions_add_up_to_the_calibrated_logit(members, candidate):
    model = _model(members, candidate)
    x, _ = _frame(400, 7)
    base, contrib = explain.contributions(model, x)
    logit = model.predict_frame(x)["logit"].to_numpy()
    assert list(contrib.columns) == [f.name for f in FEATURES]
    assert np.abs(base + contrib.to_numpy().sum(axis=1) - logit).max() < TOL


def test_platt_scales_contributions_and_shifts_the_base(members):
    x, _ = _frame(50, 3)
    base1, c1 = explain.contributions(_model(members, "ensemble", a=1.0, b=0.0), x)
    base2, c2 = explain.contributions(_model(members, "ensemble", a=0.5, b=0.3), x)
    np.testing.assert_allclose(c2.to_numpy(), 0.5 * c1.to_numpy(), atol=TOL)
    np.testing.assert_allclose(base2, 0.5 * base1 + 0.3, atol=TOL)


def test_the_ensemble_is_the_weighted_average_of_its_members(members):
    x, _ = _frame(50, 4)
    _, cg = explain.contributions(_model(members, "lgbm", a=1.0, b=0.0), x)
    _, cl = explain.contributions(_model(members, "lr", a=1.0, b=0.0), x)
    _, ce = explain.contributions(_model(members, "ensemble", a=1.0, b=0.0), x)
    np.testing.assert_allclose(ce.to_numpy(), 0.5 * (cg.to_numpy() + cl.to_numpy()), atol=TOL)


def test_lr_sums_a_categoricals_one_hot_columns_into_one_feature(members):
    _, lr, _ = members
    x, _ = _frame(30, 5)
    base, contrib = lr.contributions(x)
    assert "state" in contrib.columns and contrib["state"].abs().sum() > 0
    np.testing.assert_allclose(base + contrib.sum(axis=1), lr.logit(x), atol=TOL)


def test_the_model_uses_this_module_for_its_explanations(members):
    model = _model(members, "ensemble")
    x, _ = _frame(20, 6)
    b1, c1 = model.explain(x)
    b2, c2 = explain.contributions(model, x)
    np.testing.assert_allclose(b1, b2)
    np.testing.assert_allclose(c1.to_numpy(), c2.to_numpy())


# --------------------------------------------------------------------------- reasons

LABELS = {
    "a": ReasonLabel("citations in 24 months", "{:.0f}"),
    "b": ReasonLabel("staffing hours", "{:.2f}", "no staffing data"),
    "c": ReasonLabel("survey overdue", "yes_no"),
}


def _reason_inputs():
    contrib = pd.DataFrame({"a": [0.9, -0.1], "b": [-0.4, 0.2], "c": [0.3, -0.6], "d": [0.0, 0.5]})
    frame = pl.DataFrame(
        {
            "ccn": ["000001", "000002"],
            "t": ["2026-10", "2026-10"],
            "a": [4.0, 0.0],
            "b": [None, 3.456],
            "c": [True, False],
            "d": ["x", "y"],
        }
    )
    return contrib, frame


def test_reasons_rank_raising_and_lowering_contributions_by_size():
    contrib, frame = _reason_inputs()
    out = explain.reasons(contrib, frame, model_version=3, labels=LABELS, top_raise=2, top_lower=1)
    first = out.filter(pl.col("ccn") == "000001")
    assert first.select("direction", "rank", "feature").rows() == [
        ("raises", 1, "a"),
        ("raises", 2, "c"),
        ("lowers", 1, "b"),
    ]
    second = out.filter(pl.col("ccn") == "000002")
    assert second.select("direction", "rank", "feature").rows() == [
        ("raises", 1, "d"),
        ("raises", 2, "b"),
        ("lowers", 1, "c"),
    ]
    assert out.columns == list(explain.REASON_COLUMNS)
    assert set(out["model_version"]) == {3}


def test_a_zero_contribution_is_never_a_reason():
    contrib = pd.DataFrame({"a": [0.5], "b": [0.0], "c": [0.0]})
    frame = pl.DataFrame({"ccn": ["000001"], "t": ["2026-10"], "a": [1.0], "b": [1.0], "c": [True]})
    out = explain.reasons(contrib, frame, model_version=1, labels=LABELS, top_raise=5, top_lower=3)
    assert out.select("direction", "feature").rows() == [("raises", "a")]


def test_reason_rows_carry_labels_formatted_values_and_null_text():
    contrib, frame = _reason_inputs()
    out = explain.reasons(contrib, frame, model_version=1, labels=LABELS, top_raise=2, top_lower=1)
    row = {r["feature"]: r for r in out.filter(pl.col("ccn") == "000001").to_dicts()}
    assert row["a"]["label"] == "citations in 24 months" and row["a"]["feature_value"] == "4"
    assert row["b"]["feature_value"] == "no staffing data"  # a null with a contribution
    assert row["c"]["feature_value"] == "yes"
    second = {r["feature"]: r for r in out.filter(pl.col("ccn") == "000002").to_dicts()}
    assert second["d"]["label"] == "d" and second["d"]["feature_value"] == "y"  # no label: the name
    assert second["b"]["feature_value"] == "3.46"


def test_reasons_refuse_mismatched_rows():
    contrib, frame = _reason_inputs()
    with pytest.raises(ValueError, match="rows"):
        explain.reasons(
            contrib, frame.head(1), model_version=1, labels=LABELS, top_raise=1, top_lower=1
        )


def test_every_model_feature_has_a_written_label():
    labels = explain.load_reason_labels()
    model = [f.name for f in load_feature_registry().features if f.model_use == "model"]
    assert [f for f in model if f not in labels] == []
    raw = yaml.safe_load((project_root() / "configs" / "reason_labels.yaml").read_text())
    for name, entry in raw["labels"].items():
        fmt = entry["value_format"]
        assert fmt == "yes_no" or "{" in fmt, name


def test_an_unwritten_label_falls_back_to_the_feature_name(tmp_path):
    p = tmp_path / "labels.yaml"
    p.write_text("labels:\n  a: null\n  b: {label: B, value_format: '{:.1f}'}\n")
    labels = explain.load_reason_labels(p)
    assert explain.label_for("a", labels) == "a"
    assert explain.label_for("b", labels) == "B"
    assert explain.format_value(None, "b", labels) == explain.DEFAULT_NULL_TEXT


# --------------------------------------------------------------------------- global importance


def test_global_tables_and_key_features():
    contrib = pd.DataFrame({"n_cit": [1.0, -3.0], "hprd": [0.5, 0.5], "state": [0.0, -0.2]})
    mean_abs = explain.mean_abs_table(contrib)
    assert mean_abs["feature"].to_list() == ["n_cit", "hprd", "state"]
    assert mean_abs["mean_abs"].to_list() == pytest.approx([2.0, 0.5, 0.1])
    assert explain.key_features(mean_abs, 2) == ["n_cit", "hprd"]
    fam = explain.family_table(mean_abs, FEATURES)
    assert fam.row(0, named=True)["family"] == "fam_n"
    assert fam["share"].sum() == pytest.approx(1.0)


def test_shap_table_is_wide_with_the_base_value():
    contrib, frame = _reason_inputs()
    out = explain.shap_table(np.array([0.1, 0.2]), contrib, frame, model_version=4)
    assert out.columns[:4] == ["ccn", "t", "model_version", "base_value"]
    assert out.columns[4:] == list(contrib.columns)


def test_plots_render_png(members):
    pytest.importorskip("matplotlib")
    pytest.importorskip("shap")
    model = _model(members, "ensemble")
    x, _ = _frame(100, 8)
    _, contrib = explain.contributions(model, x)
    assert explain.bar_chart(explain.mean_abs_table(contrib)).startswith(b"\x89PNG")
    assert explain.beeswarm(contrib, x).startswith(b"\x89PNG")
