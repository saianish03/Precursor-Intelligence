"""Representation, label-rate disparity and the SliceFinder tests (bias spec §5.2-§5.4)."""

import math

import polars as pl
import pytest
from bias_fakes import make_gold
from statsmodels.stats.multitest import multipletests

from precursorintelligence.bias.disparity import label_rates, wilson_ci
from precursorintelligence.bias.representation import representation
from precursorintelligence.bias.slicefinder import (
    bh_adjust,
    cohens_h,
    find_problematic,
    two_proportion_test,
)
from precursorintelligence.bias.slices import SliceDef, build_slice_columns, enumerate_slices


def _run(cfg, df):
    df = build_slice_columns(df, cfg)
    slices = enumerate_slices(df, cfg)
    rep = representation(df, slices, cfg)
    rates = label_rates(df, slices, cfg, ["y_g12"])
    return df, rep, rates, find_problematic(rates, rep, cfg)


def _row(df, **eq):
    out = df
    for k, v in eq.items():
        out = out.filter(pl.col(k) == v)
    assert out.height == 1, out
    return out.row(0, named=True)


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(50, 100, 0.95)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson_ci(0, 0, 0.95) == (None, None)


def test_cohens_h():
    assert cohens_h(0.5, 0.3) == pytest.approx(0.411, abs=1e-3)
    assert cohens_h(0.3, 0.3) == 0


def test_bh_fdr_known_pvalues():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    assert bh_adjust(p) == pytest.approx(list(multipletests(p, method="fdr_bh")[1]))
    assert bh_adjust([]) == []


def test_fisher_fallback_small_counts():
    p, test = two_proportion_test(3, 10, 30, 100)
    assert test == "fisher"
    assert 0 < p <= 1
    _, test = two_proportion_test(30, 100, 60, 100)
    assert test == "z"


def test_only_labeled_rows_used(cfg):
    df = make_gold(n_fac=30, months=4).with_columns(
        # every row of month 2024-04 is immature, and of 2024-03 censored: excluded but counted
        pl.when(pl.col("t") == "2024-04")
        .then(pl.lit("immature"))
        .when(pl.col("t") == "2024-03")
        .then(pl.lit("censored"))
        .otherwise(pl.col("status_g12"))
        .alias("status_g12"),
        pl.when(pl.col("t").is_in(["2024-03", "2024-04"]))
        .then(None)
        .otherwise(pl.col("y_g12"))
        .alias("y_g12"),
    )
    _, _, rates, _ = _run(cfg, df)
    r = _row(rates, slice="ownership_group", value="for_profit")
    assert r["n"] == 10 * 2  # 10 homes × the 2 labeled months
    assert r["n_immature"] == 10
    assert r["n_censored"] == 10


def test_ratio_and_complement(cfg):
    df = make_gold(rate={"government": 0.6, "for_profit": 0.2, "non_profit": 0.2})
    _, _, rates, _ = _run(cfg, df)
    r = _row(rates, slice="ownership_group", value="government")
    # 1200 government rows at 0.6, 2400 others at 0.2
    assert (r["n"], r["positives"]) == (1200, 720)
    assert (r["complement_n"], r["complement_positives"]) == (2400, 480)
    assert r["rate"] == pytest.approx(0.6)
    assert r["complement_rate"] == pytest.approx(0.2)
    assert r["ratio"] == pytest.approx(3.0)
    assert r["diff"] == pytest.approx(0.4)
    assert r["flag_ratio"] is True
    assert r["ci_low"] < 0.6 < r["ci_high"]


def test_planted_disparity_found(cfg):
    df = make_gold(rate={"government": 0.6, "for_profit": 0.2, "non_profit": 0.2})
    _, _, _, ranked = _run(cfg, df)
    top = ranked.filter(pl.col("rank") == 1).row(0, named=True)
    assert (top["slice"], top["value"]) == ("ownership_group", "government")
    assert top["problematic"] is True
    assert top["adj_p"] < cfg.thresholds.fdr_alpha


def test_no_disparity_no_flags(cfg):
    df = make_gold(rate={"government": 0.2, "for_profit": 0.2, "non_profit": 0.2})
    _, _, rates, ranked = _run(cfg, df)
    assert not ranked["problematic"].any()
    assert not rates.filter(pl.col("slice") == "ownership_group")["flag_ratio"].any()


def test_small_slice_never_problematic(cfg):
    # one home with a 100% rate among 299 at 0
    df = make_gold(rate={"for_profit": 0.0, "non_profit": 0.0, "government": 0.0})
    df = df.with_columns(
        pl.when(pl.col("ccn") == "100000")
        .then(pl.lit("VT"))
        .otherwise(pl.col("state"))
        .alias("state"),
        pl.when(pl.col("ccn") == "100000")
        .then(1)
        .otherwise(pl.col("y_g12"))
        .cast(pl.Int8)
        .alias("y_g12"),
    ).with_columns(
        pl.when(pl.col("y_g12") == 1)
        .then(pl.lit("positive"))
        .otherwise(pl.lit("negative"))
        .alias("status_g12")
    )
    _, rep, _, ranked = _run(cfg, df)
    assert _row(rep, slice="state", value="VT")["small_slice"] is True
    vt = _row(ranked, slice="state", value="VT")
    assert abs(vt["cohens_h"]) > 1
    assert vt["problematic"] is False


def test_empty_slice_value(cfg, gold_frame):
    # no home has fewer than 50 beds after this, and no row is in CMS region 2
    df = gold_frame.with_columns(pl.col("certified_beds").clip(lower_bound=60))
    _, rep, rates, ranked = _run(cfg, df)
    r = _row(rep, slice="bed_band", value="<50")
    assert (r["n_rows"], r["n_facilities"], r["small_slice"]) == (0, 0, True)
    r = _row(rates, slice="bed_band", value="<50")
    assert r["n"] == 0
    assert r["rate"] is None and r["flag_ratio"] is False
    assert ranked.filter((pl.col("slice") == "bed_band") & (pl.col("value") == "<50")).height == 0


def test_representation_shares(cfg, gold_frame):
    df = build_slice_columns(gold_frame, cfg)
    rep = representation(df, [SliceDef(("ownership_group",), ("government",))], cfg)
    r = rep.row(0, named=True)
    assert (r["n_rows"], r["n_facilities"]) == (1200, 100)
    assert r["share_rows"] == pytest.approx(1 / 3)
    assert (r["first_t"], r["last_t"]) == ("2024-01", "2024-12")
    assert not math.isnan(r["share_facilities"])
