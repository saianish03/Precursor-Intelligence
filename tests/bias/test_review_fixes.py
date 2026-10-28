"""Fixes from the PR #26 review: home-level inference (M1), train-only weights (M2), the chain_id era (M3),
"none" footnotes (M4), model features only (M5), per-label scope rows (M6), config and value guards (S7)."""

import numpy as np
import polars as pl
import pytest
from bias_fakes import make_gold
from pydantic import ValidationError

from precursorintelligence.bias.config import BiasConfig
from precursorintelligence.bias.disparity import clustered_variance, label_rates
from precursorintelligence.bias.io import GoldData
from precursorintelligence.bias.missingness import missingness_by_slice, scan_features
from precursorintelligence.bias.mitigation import sample_weights
from precursorintelligence.bias.report import run_bias_report
from precursorintelligence.bias.slicefinder import find_problematic, two_proportion_test
from precursorintelligence.bias.slices import (
    MISSING,
    build_slice_columns,
    enumerate_slices,
    slice_col,
)
from precursorintelligence.common.config import load_feature_registry

STATES = ["MA", "CA", "TX", "NY", "FL", "OH", "WA", "GA"]


@pytest.fixture
def registry():
    return load_feature_registry()


def _only(cfg, single, crossings=()):
    return cfg.model_copy(
        update={"slices": cfg.slices.model_copy(update={"single": single, "crossings": crossings})}
    )


def _persistent_homes(seed: int, n_homes: int = 400, months: int = 24) -> pl.DataFrame:
    """Each home is high- or low-risk for every month (labels strongly correlated within a home); its state
    is drawn independently of risk, so no state truly differs."""
    rng = np.random.default_rng(seed)
    risk = np.where(rng.random(n_homes) < 0.3, 0.7, 0.1)
    state = rng.choice(STATES, n_homes)
    rows = []
    for h in range(n_homes):
        for m in range(months):
            y = int(rng.random() < risk[h]) if m % 6 == 0 else None  # re-drawn twice a year
            rows.append((f"{100000 + h:06d}", f"t{m:02d}", state[h], y))
    df = pl.DataFrame(rows, schema=["ccn", "t", "state", "_y"], orient="row", strict=False)
    # carry each draw forward 6 months: overlapping windows keep a home's label for months at a time
    df = df.with_columns(pl.col("_y").forward_fill().over("ccn").cast(pl.Int8).alias("y_g12"))
    return df.drop("_y").with_columns(
        pl.when(pl.col("y_g12") == 1)
        .then(pl.lit("positive"))
        .otherwise(pl.lit("negative"))
        .alias("status_g12")
    )


# --------------------------------------------------------------------------- M1


def test_clustered_variance_matches_brute_force():
    rng = np.random.default_rng(1)
    rows = []
    for h in range(60):
        p = rng.uniform(0.05, 0.6)
        for _ in range(int(rng.integers(1, 10))):
            g = (
                rng.choice(["a", "b"]) if h % 5 == 0 else ["a", "b"][h % 2]
            )  # some homes switch value
            rows.append((f"h{h}", g, int(rng.random() < p)))
    df = pl.DataFrame(rows, schema=["ccn", slice_col("g"), "y"], orient="row")
    got = clustered_variance(df, "y", ("g",)).filter(pl.col("value") == "a").row(0, named=True)
    ccn, g, y = (
        df["ccn"].to_numpy(),
        df[slice_col("g")].to_numpy(),
        df["y"].to_numpy().astype(float),
    )
    s = g == "a"
    n1, n2, p1, p2 = s.sum(), (~s).sum(), y[s].mean(), y[~s].mean()
    e = np.where(s, (y - p1) / n1, -(y - p2) / n2)
    homes = np.unique(ccn)
    u = np.array([e[ccn == c].sum() for c in homes])
    assert got["var_diff"] == pytest.approx((u**2).sum() * len(homes) / (len(homes) - 1))
    hs = np.unique(ccn[s])
    a = np.array([(y[s & (ccn == c)] - p1).sum() for c in hs]) / n1
    assert got["var_rate"] == pytest.approx((a**2).sum() * len(hs) / (len(hs) - 1))
    assert got["n_homes"] == len(hs)


def test_null_with_correlated_homes_stays_near_nominal(cfg):
    """No true state difference, labels persistent within a home: the clustered test rejects near 5%,
    where the row-level test (independent home-months) rejects far more often."""
    cfg = _only(cfg, ["state"])
    clustered, naive, tests = 0, 0, 0
    for seed in range(12):
        df = build_slice_columns(_persistent_homes(seed), cfg)
        rates = label_rates(df, enumerate_slices(df, cfg), cfg, ["y_g12"])
        ranked = find_problematic(rates, cfg)
        clustered += int((ranked["p_value"] < 0.05).sum())
        for r in rates.filter(pl.col("n") > 0).iter_rows(named=True):
            p, _ = two_proportion_test(
                r["positives"], r["n"], r["complement_positives"], r["complement_n"]
            )
            naive += int(p < 0.05)
        tests += ranked.height
    assert tests == 12 * len(STATES)
    assert clustered / tests < 0.12
    assert naive / tests > 0.25


def test_wilson_interval_sized_on_homes(cfg):
    cfg = _only(cfg, ["state"])
    df = build_slice_columns(_persistent_homes(0), cfg)
    rates = label_rates(df, enumerate_slices(df, cfg), cfg, ["y_g12"]).filter(pl.col("n") > 0)
    for r in rates.iter_rows(named=True):
        assert r["n_homes"] <= r["n_eff"] < r["n"]  # correlated months add less than a home each
        assert r["ci_low"] < r["rate"] < r["ci_high"]


def test_planted_disparity_still_found(cfg):
    df = build_slice_columns(make_gold(rate={"government": 0.6}), cfg)
    ranked = find_problematic(label_rates(df, enumerate_slices(df, cfg), cfg, ["y_g12"]), cfg)
    top = ranked.filter(pl.col("rank") == 1).row(0, named=True)
    assert (top["slice"], top["value"], top["test_used"]) == (
        "ownership_group",
        "government",
        "z_cluster",
    )
    assert top["problematic"] is True


def test_small_slice_counts_labeled_rows(cfg):
    """A slice with many rows but few labeled ones is small for that label."""
    df = make_gold(n_fac=300, months=12).with_columns(
        pl.when((pl.col("ownership_group") == "government") & (pl.col("t") != "2024-01"))
        .then(pl.lit("immature"))
        .otherwise(pl.col("status_g12"))
        .alias("status_g12")
    )
    df = build_slice_columns(df, cfg)
    ranked = find_problematic(label_rates(df, enumerate_slices(df, cfg), cfg, ["y_g12"]), cfg)
    gov = ranked.filter((pl.col("slice") == "ownership_group") & (pl.col("value") == "government"))
    assert gov.row(0, named=True)["n"] == 100  # 100 homes × 1 labeled month < min_slice_rows
    assert gov.row(0, named=True)["small_slice"] is True


# --------------------------------------------------------------------------- M2


@pytest.mark.parametrize("scheme", ["inverse_frequency", "balance_label_rate"])
def test_weights_ignore_held_out_rows(cfg, scheme):
    df = build_slice_columns(make_gold(rate={"government": 0.6, "non_profit": 0.3}), cfg)
    held_out = pl.col("split_g12") != "train"
    flipped = df.with_columns(
        pl.when(held_out).then(1 - pl.col("y_g12")).otherwise(pl.col("y_g12")).alias("y_g12"),
        # and move held-out homes to another ownership group
        pl.when(held_out)
        .then(pl.lit("government"))
        .otherwise(pl.col(slice_col("ownership_group")))
        .alias(slice_col("ownership_group")),
    )
    a, b = sample_weights(df, cfg, scheme), sample_weights(flipped, cfg, scheme)
    assert a.height == df.height
    assert a["weight"].equals(b["weight"])
    is_train = (df["split_g12"] == "train").to_list()
    assert all(w is None for w, tr in zip(a["weight"].to_list(), is_train, strict=True) if not tr)
    assert a.filter(pl.Series(is_train))["weight"].null_count() == 0


def test_weights_mean_one_after_clipping(cfg):
    df = build_slice_columns(make_gold(), cfg).filter(
        (pl.col("ownership_group") == "for_profit") | (pl.col("ccn").str.slice(-2) == "02")
    )
    tight = cfg.model_copy(
        update={"mitigation": cfg.mitigation.model_copy(update={"clip": (0.5, 2.0)})}
    )
    w = sample_weights(df, tight, "inverse_frequency")["weight"].drop_nulls()
    assert w.mean() == pytest.approx(1.0)
    assert w.min() >= 0.5 - 1e-12 and w.max() <= 2.0 + 1e-12
    assert w.max() == pytest.approx(2.0)  # the clip binds for the rare groups


def test_weights_without_split_are_null(cfg):
    df = build_slice_columns(make_gold(split=False), cfg)
    assert sample_weights(df, cfg)["weight"].null_count() == df.height


# --------------------------------------------------------------------------- M3


def test_chain_status_missing_before_chain_id_exists(cfg, registry):
    df = pl.DataFrame(
        {
            "t": ["2023-05", "2023-05", "2023-06", "2023-06"],
            "state": ["MA"] * 4,
            "ownership_group": ["for_profit"] * 4,
            "certified_beds": [100.0] * 4,
            "chain_id": [None, "C1", None, "C1"],
            "provider_in_hospital": [False] * 4,
            "ccrc": [False] * 4,
            "sff_status": ["NONE"] * 4,
            "overall_rating": [3.0] * 4,
            "era": ["pre_2025_07"] * 4,
        }
    )
    out = build_slice_columns(df, cfg, registry)[slice_col("chain_status")].to_list()
    assert out == [MISSING, MISSING, "independent", "chain"]
    # without the registry (no era information) a null chain_id still means independent
    assert build_slice_columns(df, cfg)[slice_col("chain_status")][0] == "independent"


# --------------------------------------------------------------------------- M4, M5


def test_footnote_none_is_no_reason(cfg, gold_frame):
    df = build_slice_columns(gold_frame, cfg).with_row_index("i")
    df = df.with_columns(
        pl.when((pl.col("ownership_group") == "government") & (pl.col("i") % 5 < 2))
        .then(pl.lit("not_submitted"))
        .otherwise(pl.col("turnover_total_fn"))
        .alias("turnover_total_fn")
    )
    out = missingness_by_slice(df, ["ownership_group"], ["turnover_total_fn"], cfg)
    gov = out.filter(pl.col("value") == "government").row(0, named=True)
    assert gov["kind"] == "has_reason"
    assert gov["null_rate"] == pytest.approx(0.4, abs=0.01)
    assert gov["overall_null_rate"] == pytest.approx(0.4 / 3, abs=0.01)
    assert gov["flag"] is True
    assert out.filter(pl.col("value") == "for_profit")["null_rate"][0] == 0


def test_only_model_features_scanned(gold_frame, registry):
    features = scan_features(gold_frame, registry)
    assert "hprd_total" in features and "turnover_total_fn" in features
    assert "chain_id" not in features and "overall_rating" not in features


# --------------------------------------------------------------------------- M6


def test_each_label_uses_its_own_train_rows(cfg, registry, tmp_path):
    """y_g12 trains on the first half of the months, y_pen12 on the first quarter only."""
    df = make_gold(n_fac=300, months=12).with_columns(
        pl.when(pl.col("t") <= "2024-03")
        .then(pl.lit("train"))
        .otherwise(pl.lit("test"))
        .alias("split_pen12")
    )
    loose = cfg.model_copy(
        update={"thresholds": cfg.thresholds.model_copy(update={"min_slice_rows": 100})}
    )
    (tmp_path / "cfg.yaml").write_text("x", encoding="utf-8")
    res = run_bias_report(
        GoldData(df=df, source="fake", pointer={}),
        loose,
        labels=["y_pen12", "y_g12"],
        scope="train",
        out_dir=tmp_path / "out",
        registry=registry,
        config_path=tmp_path / "cfg.yaml",
        charts=False,
    )
    rater = res.tables["rater_bias_by_state.csv"]
    n = dict(rater.group_by("label").agg(pl.col("n").sum()).iter_rows())
    assert n == {"y_pen12": 300 * 3, "y_g12": 300 * 6}
    assert res.summary["rows_in_scope"] == 300 * 6  # label-free checks: train for either label
    assert res.summary["label_rows"]["y_pen12"]["used"] == 300 * 3


def test_label_without_train_rows_does_not_stop_the_others(cfg, registry, tmp_path):
    df = make_gold(n_fac=90, months=8).with_columns(pl.lit("test").alias("split_pen12"))
    (tmp_path / "cfg.yaml").write_text("x", encoding="utf-8")
    res = run_bias_report(
        GoldData(df=df, source="fake", pointer={}),
        cfg,
        labels=["y_pen12", "y_g12"],
        scope="train",
        out_dir=tmp_path / "out",
        registry=registry,
        config_path=tmp_path / "cfg.yaml",
        charts=False,
    )
    assert res.summary["label_rows"]["y_pen12"]["used"] == 0
    assert res.summary["label_rows"]["y_g12"]["used"] == 90 * 4


# --------------------------------------------------------------------------- S4, S7


def test_config_rejects_self_crossing_and_non_feature_weight_slice(cfg):
    raw = cfg.model_dump(by_alias=True)
    with pytest.raises(ValidationError, match="crossed with itself"):
        BiasConfig.model_validate(
            raw | {"slices": {"single": ["state"], "crossings": [["state", "state"]]}}
        )
    for bad in ("star_band", "chain_status"):
        with pytest.raises(ValidationError, match="not a model"):
            BiasConfig.model_validate(
                raw | {"mitigation": raw["mitigation"] | {"weight_slice": bad}}
            )


def test_nan_rating_and_zero_beds_are_missing(cfg):
    df = make_gold(n_fac=3, months=1).with_columns(
        pl.Series("overall_rating", [float("nan"), 2.0, None]),
        pl.Series("certified_beds", [0.0, -5.0, 10.0]),
    )
    out = build_slice_columns(df, cfg)
    assert out[slice_col("star_band")].to_list() == [MISSING, "2", MISSING]
    assert out[slice_col("bed_band")].to_list() == [MISSING, MISSING, "<50"]
