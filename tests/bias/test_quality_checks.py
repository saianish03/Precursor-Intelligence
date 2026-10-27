"""Missingness, skew, rater bias and the mitigation weights (bias spec §5.5-§5.8)."""

import numpy as np
import polars as pl
import pytest
from bias_fakes import make_gold

from precursorintelligence.bias.disparity import labeled
from precursorintelligence.bias.missingness import missingness_by_slice
from precursorintelligence.bias.mitigation import effective_sample_size, sample_weights
from precursorintelligence.bias.rater_bias import rater_bias_by_state
from precursorintelligence.bias.skew import (
    TIME_THIRDS,
    js_distance,
    linf_distance,
    numeric_js,
    skew_by_slice,
)
from precursorintelligence.bias.slices import build_slice_columns, slice_col


def test_missingness_gap(cfg, gold_frame):
    # 40% of government rows lose hprd_total; nothing else is missing
    df = build_slice_columns(gold_frame, cfg).with_row_index("i")
    df = df.with_columns(
        pl.when((pl.col("ownership_group") == "government") & (pl.col("i") % 5 < 2))
        .then(None)
        .otherwise(pl.col("hprd_total"))
        .alias("hprd_total")
    )
    out = missingness_by_slice(df, ["ownership_group"], ["hprd_total", "turnover_total_fn"], cfg)
    gov = out.filter((pl.col("value") == "government") & (pl.col("feature") == "hprd_total")).row(
        0, named=True
    )
    assert gov["null_rate"] == pytest.approx(0.4, abs=0.01)
    assert gov["overall_null_rate"] == pytest.approx(0.4 / 3, abs=0.01)
    assert gov["flag"] is True
    assert not out.filter(pl.col("value") == "for_profit")["flag"].any()
    fn = out.filter(pl.col("feature") == "turnover_total_fn")
    assert set(fn["kind"]) == {"has_reason"} and not fn["flag"].any()


def test_js_and_linf():
    assert js_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0)
    assert js_distance([1, 0], [0, 1]) == pytest.approx(1)
    a = np.arange(100, dtype=float)
    assert numeric_js(a, a.copy(), 20) == pytest.approx(0)
    assert numeric_js(a, a + 1000, 20) == pytest.approx(1)  # disjoint: all in the overflow bin
    assert linf_distance(["x", "y"], ["x", "y"]) == 0
    assert linf_distance(["x", "x"], ["y", "y"]) == 1
    assert linf_distance(["x", "y", "y", "y"], ["x", "x", "x", "y"]) == pytest.approx(0.5)


def _loose(cfg):
    """min_slice_rows 100: the synthetic test split has 300 rows per ownership group."""
    return cfg.model_copy(
        update={"thresholds": cfg.thresholds.model_copy(update={"min_slice_rows": 100})}
    )


def test_skew_uses_splits(cfg, gold_frame):
    cfg = _loose(cfg)
    df = build_slice_columns(gold_frame, cfg)
    out, method = skew_by_slice(df, ["ownership_group"], ["hprd_total", "ccrc"], cfg)
    assert method.startswith("split_g12")
    assert set(out["metric"]) == {"js", "linf"}
    hprd = out.filter(pl.col("feature") == "hprd_total")
    assert (hprd["distance"] == 0).all()  # hprd_total does not change over time
    assert (out["n_ref"] == 600).all() and (out["n_cmp"] == 300).all()


def test_skew_fallback_without_splits(cfg):
    cfg = _loose(cfg)
    df = build_slice_columns(make_gold(split=False), cfg)
    out, method = skew_by_slice(df, ["ownership_group"], ["days_since_last_std"], cfg)
    assert method.startswith(TIME_THIRDS)
    assert "2024-01..2024-04 vs 2024-09..2024-12" in method
    assert out.height == 3 and (out["n_ref"] == 400).all()


def test_rater_bias_flags_state(cfg):
    df = make_gold(n_fac=300)
    # MA: more inspections and a higher label rate
    df = df.with_columns(
        pl.when(pl.col("state") == "MA").then(9).otherwise(pl.col("n_std_36m")).alias("n_std_36m"),
        pl.when(pl.col("state") == "MA")
        .then(1)
        .otherwise(pl.col("y_g12"))
        .cast(pl.Int8)
        .alias("y_g12"),
    )
    table, summary = rater_bias_by_state(df, cfg, ["y_g12"])
    assert summary["y_g12"]["flagged_states"] == ["MA"]
    ma = table.filter(pl.col("state") == "MA").row(0, named=True)
    assert ma["label_rate"] == 1.0 and ma["mean_n_std_36m"] == 9
    assert summary["y_g12"]["spearman"]["n_std_36m"]["rho"] > 0


def test_weights_inverse_frequency(cfg):
    # for_profit 3 times as common as government
    df = make_gold(n_fac=300).filter(
        (pl.col("ownership_group") != "non_profit")
        & (
            (pl.col("ownership_group") == "for_profit")
            | (pl.col("ccn").str.slice(-1).cast(pl.Int32) % 3 == 0)
        )
    )
    df = build_slice_columns(df, cfg)
    w = sample_weights(df, cfg, "inverse_frequency")
    assert w.height == df.height and w.columns == ["ccn", "t", "weight", "scheme"]
    assert w["weight"].null_count() == df.filter(pl.col("split_g12") != "train").height
    assert w["weight"].mean() == pytest.approx(1.0)
    by = (
        w.with_columns(df[slice_col("ownership_group")].alias("g"))
        .group_by("g")
        .agg(pl.col("weight").drop_nulls().first())
    )
    weight = dict(by.iter_rows())
    assert weight["government"] > weight["for_profit"]
    lo, hi = cfg.mitigation.clip
    extreme = cfg.model_copy(
        update={"mitigation": cfg.mitigation.model_copy(update={"clip": (0.9, 1.1)})}
    )
    clipped = sample_weights(df, extreme, "inverse_frequency")["weight"]
    assert clipped.min() >= 0.9 and clipped.max() <= 1.1
    assert lo <= w["weight"].min() and w["weight"].max() <= hi


def test_weights_balance_label_rate(cfg):
    df = make_gold(rate={"government": 0.6, "for_profit": 0.2, "non_profit": 0.3})
    wide = cfg.model_copy(
        update={"mitigation": cfg.mitigation.model_copy(update={"clip": (0.01, 100.0)})}
    )
    df = build_slice_columns(df, wide)
    w = sample_weights(df, wide, "balance_label_rate")
    train = pl.col("split_g12") == "train"
    lab = labeled(df.filter(train), wide, "y_g12")
    overall = lab["y_g12"].cast(pl.Float64).mean()
    joined = (
        df.select(slice_col("ownership_group"), "y_g12", train.alias("_train"))
        .with_columns(w["weight"])
        .filter(pl.col("_train"))
    )
    rates = joined.group_by(slice_col("ownership_group")).agg(
        ((pl.col("weight") * pl.col("y_g12")).sum() / pl.col("weight").sum()).alias("r")
    )
    for r in rates["r"]:
        assert r == pytest.approx(overall, abs=1e-9)
    assert w["weight"].mean() == pytest.approx(1.0)


def test_effective_sample_size():
    assert effective_sample_size(pl.Series([1.0, 1.0, 1.0, 1.0])) == pytest.approx(4)
    assert effective_sample_size(pl.Series([4.0, 0.0, 0.0, 0.0])) == pytest.approx(1)
