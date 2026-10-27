"""One test per mutation that survived the PR #26 review's mutation run: each pins a one-line bug no other
test caught (inputs and expected values from the review, section 6)."""

import math

import numpy as np
import polars as pl
import pytest
from bias_fakes import make_gold
from statsmodels.stats.proportion import proportions_ztest

from precursorintelligence.bias.disparity import label_rates, labeled, status_totals
from precursorintelligence.bias.io import apply_scope
from precursorintelligence.bias.missingness import missingness_by_slice
from precursorintelligence.bias.mitigation import sample_weights
from precursorintelligence.bias.rater_bias import rater_bias_by_state
from precursorintelligence.bias.representation import representation
from precursorintelligence.bias.skew import js_distance, numeric_js, skew_by_slice
from precursorintelligence.bias.slicefinder import (
    cohens_h,
    find_problematic,
    two_proportion_test,
)
from precursorintelligence.bias.slices import (
    SliceDef,
    build_slice_columns,
    enumerate_slices,
    slice_col,
)


def _only(cfg, single, crossings=()):
    return cfg.model_copy(
        update={"slices": cfg.slices.model_copy(update={"single": single, "crossings": crossings})}
    )


def _rates_row(value, n, x, cn, cx, *, homes=100, var_diff=None):
    """One label_rates-shaped row of ownership_group = value, for find_problematic."""
    rate, crate = x / n, cx / cn
    return {
        "label": "y_g12",
        "slice": "ownership_group",
        "value": value,
        "n": n,
        "n_homes": homes,
        "positives": x,
        "rate": rate,
        "complement_n": cn,
        "complement_positives": cx,
        "complement_rate": crate,
        "var_diff": var_diff,
    }


def _labels(df, y, status=None):
    return df.with_columns(
        pl.Series("y_g12", y, dtype=pl.Int8),
        pl.Series(
            "status_g12",
            status or ["positive" if v == 1 else "negative" for v in y],
            dtype=pl.String,
        ),
    )


# --------------------------------------------------------------------------- the tests themselves


def test_fisher_only_below_five():
    assert two_proportion_test(5, 100, 50, 100)[1] == "z"  # smallest cell is exactly 5


def test_z_test_pooled_and_two_sided():
    p, test = two_proportion_test(30, 100, 600, 4000)
    _, ref = proportions_ztest([30, 600], [100, 4000])  # pooled, two-sided
    assert test == "z"
    assert p == pytest.approx(ref) and p == pytest.approx(3.98e-05, rel=0.01)


def test_fisher_two_sided():
    p, test = two_proportion_test(3, 10, 30, 100)
    assert test == "fisher" and p == pytest.approx(1.0)


def test_problematic_uses_adjusted_p(cfg):
    """Raw p 0.025 < 0.05, but BH over 3 tests makes it 0.075: not problematic."""
    z = 2.2414  # two-sided p = 0.025
    rows = [
        _rates_row("government", 100 * 6, 35 * 6, 2000, 500, var_diff=(0.1 / z) ** 2),
        _rates_row("for_profit", 600, 150, 2000, 500, var_diff=0.01),
        _rates_row("non_profit", 600, 150, 2000, 500, var_diff=0.01),
    ]
    ranked = find_problematic(pl.DataFrame(rows), cfg)
    gov = ranked.filter(pl.col("value") == "government").row(0, named=True)
    assert gov["cohens_h"] == pytest.approx(0.219, abs=1e-3)
    assert gov["p_value"] == pytest.approx(0.025, abs=1e-3)
    assert gov["adj_p"] == pytest.approx(0.075, abs=1e-3)
    assert gov["problematic"] is False


def test_problematic_ranked_before_a_larger_small_slice(cfg):
    rows = [
        _rates_row("government", 10, 10, 5000, 1000, homes=1),  # one home at 1.0: h ~ 2.2, small
        _rates_row("for_profit", 3000, 1500, 3000, 600, var_diff=1e-4),  # 0.5 vs 0.2: problematic
    ]
    ranked = find_problematic(pl.DataFrame(rows), cfg).sort("rank")
    assert abs(ranked["cohens_h"][1]) > abs(ranked["cohens_h"][0])
    assert ranked["value"].to_list() == ["for_profit", "government"]
    assert ranked["problematic"].to_list() == [True, False]


def test_labeled_needs_status_and_label(cfg):
    df = pl.DataFrame(
        {
            "y_g12": pl.Series([1, 1, None, 0], dtype=pl.Int8),
            "status_g12": ["immature", "positive", "positive", "negative"],
        }
    )
    assert labeled(df, cfg, "y_g12")["status_g12"].to_list() == ["positive", "negative"]


def test_positives_rounded_not_truncated(cfg):
    cfg = _only(cfg, ["ownership_group"])
    df = make_gold(n_fac=22, months=1)
    df = _labels(df.with_columns(pl.lit("for_profit").alias("ownership_group")), [1] * 15 + [0] * 7)
    df = build_slice_columns(df, cfg)
    rates = label_rates(df, [SliceDef(("ownership_group",), ("for_profit",))], cfg, ["y_g12"])
    assert rates.row(0, named=True)["positives"] == 15  # 15/22 * 22 = 14.999...


def test_ratio_exactly_at_the_bounds_not_flagged(cfg):
    cfg = _only(cfg, ["ownership_group"])
    df = make_gold(n_fac=16, months=1).with_columns(
        pl.Series("ownership_group", ["for_profit"] * 8 + ["non_profit"] * 8)
    )
    df = build_slice_columns(_labels(df, [1] * 5 + [0] * 3 + [1] * 4 + [0] * 4), cfg)
    rates = label_rates(df, enumerate_slices(df, cfg), cfg, ["y_g12"]).filter(pl.col("n") > 0)
    by = {r["value"]: r for r in rates.iter_rows(named=True)}
    assert by["for_profit"]["ratio"] == 1.25 and by["non_profit"]["ratio"] == 0.8  # 5/8 vs 4/8
    assert not rates["flag_ratio"].any()


def test_complement_at_zero_is_flagged(cfg):
    cfg = _only(cfg, ["ownership_group"])
    df = make_gold(n_fac=16, months=1).with_columns(
        pl.Series("ownership_group", ["for_profit"] * 8 + ["non_profit"] * 8)
    )
    df = build_slice_columns(_labels(df, [1] * 8 + [0] * 8), cfg)
    rates = label_rates(df, enumerate_slices(df, cfg), cfg, ["y_g12"])
    fp = rates.filter(pl.col("value") == "for_profit").row(0, named=True)
    assert fp["rate"] == 1.0 and fp["complement_rate"] == 0.0
    assert fp["ratio"] is None and fp["flag_ratio"] is True


def test_blank_chain_id_is_independent(cfg):
    df = make_gold(n_fac=2, months=1).with_columns(pl.Series("chain_id", ["  ", "C9"]))
    assert build_slice_columns(df, cfg)[slice_col("chain_status")].to_list() == [
        "independent",
        "chain",
    ]


def test_unlabeled_train_rows_get_no_weight(cfg):
    df = make_gold(n_fac=30, months=4).with_columns(
        pl.when(pl.col("t") == "2024-01")
        .then(pl.lit("immature"))
        .otherwise(pl.col("status_g12"))
        .alias("status_g12")
    )
    df = build_slice_columns(df, cfg)
    w = sample_weights(df, cfg, "balance_label_rate").with_columns(df["t"].alias("_t"))
    assert w.filter(pl.col("_t") == "2024-01")["weight"].null_count() == 30
    assert w.filter(pl.col("_t") == "2024-02")["weight"].null_count() == 0  # labeled train month


def test_js_distance_not_divergence_and_reference_bins():
    assert js_distance([1, 1], [1, 0]) == pytest.approx(0.5579, abs=1e-4)
    assert numeric_js(np.arange(100.0), np.arange(50.0), 20) == pytest.approx(0.5579, abs=1e-4)


def test_skew_skips_slice_small_in_test(cfg):
    loose = cfg.model_copy(
        update={"thresholds": cfg.thresholds.model_copy(update={"min_slice_rows": 100})}
    )
    df = build_slice_columns(make_gold(), loose)
    # government keeps only 50 test rows (300 before)
    df = df.filter(
        ~(
            (pl.col("ownership_group") == "government")
            & (pl.col("split_g12") == "test")
            & (pl.col("ccn").str.slice(-1).cast(pl.Int32) > 1)
        )
    )
    out, _ = skew_by_slice(df, ["ownership_group"], ["hprd_total"], loose)
    assert set(out["value"]) == {"for_profit", "non_profit"}


def _states(n_states=10, homes=30, months=2):
    rows = []
    for s in range(n_states):
        for h in range(homes):
            for m in range(months):
                rows.append(
                    {
                        "ccn": f"{s:02d}{h:04d}",
                        "t": f"2024-0{m + 1}",
                        "state": f"S{s}",
                        "n_std_36m": 2.0,
                        "days_since_last_std": 100.0,
                        "std_overdue": False,
                        "state_gplus_rate_12m": 0.1,
                        "y_g12": 0,
                        "status_g12": "negative",
                    }
                )
    return pl.DataFrame(rows).with_columns(pl.col("y_g12").cast(pl.Int8))


def test_rater_rate_on_labeled_rows_and_mean_exposure(cfg):
    """S0: label rate 1.0 on its labeled month (the other month is immature, y = 0); its mean exposure is high
    because 3 of 30 homes inspect a lot, while its median is ordinary."""
    df = _states().with_columns(
        pl.when(pl.col("state") == "S0")
        .then(pl.when(pl.col("t") == "2024-01").then(1).otherwise(0))
        .otherwise(pl.col("y_g12"))
        .cast(pl.Int8)
        .alias("y_g12"),
        pl.when((pl.col("state") == "S0") & (pl.col("t") == "2024-02"))
        .then(pl.lit("immature"))
        .when(pl.col("state") == "S0")
        .then(pl.lit("positive"))
        .otherwise(pl.col("status_g12"))
        .alias("status_g12"),
        pl.when((pl.col("state") == "S0") & (pl.col("ccn").str.slice(-4).cast(pl.Int32) < 3))
        .then(60.0)
        .otherwise(pl.col("n_std_36m"))
        .alias("n_std_36m"),
    )
    table, summary = rater_bias_by_state(df, cfg, ["y_g12"])
    s0 = table.filter(pl.col("state") == "S0").row(0, named=True)
    assert s0["label_rate"] == 1.0 and s0["n"] == 30
    assert s0["median_n_std_36m"] == 2.0 and s0["mean_n_std_36m"] > 2.0
    assert summary["y_g12"]["flagged_states"] == ["S0"]


def test_rater_small_states_not_ranked(cfg):
    df = _states(homes=30).vstack(
        _states(n_states=1, homes=2).with_columns(
            pl.lit("GU").alias("state"),
            pl.lit(1).cast(pl.Int8).alias("y_g12"),
            pl.lit("positive").alias("status_g12"),
            pl.lit(50.0).alias("n_std_36m"),
            (pl.lit("GU") + pl.col("ccn")).alias("ccn"),
        )
    )
    table, summary = rater_bias_by_state(df, cfg, ["y_g12"])
    gu = table.filter(pl.col("state") == "GU").row(0, named=True)
    assert gu["eligible"] is False and gu["flag"] is False
    assert summary["y_g12"]["states_ranked"] == 10


def test_missingness_codes_and_nan(cfg, gold_frame):
    df = build_slice_columns(gold_frame, cfg).with_row_index("i")
    gov40 = (pl.col("ownership_group") == "government") & (pl.col("i") % 5 < 2)
    df = df.with_columns(
        pl.when(gov40)
        .then(pl.lit("not_submitted"))
        .otherwise(pl.col("turnover_total_fn"))
        .alias("turnover_total_fn"),
        pl.when(gov40).then(float("nan")).otherwise(pl.col("hprd_total")).alias("hprd_total"),
    )
    out = missingness_by_slice(df, ["ownership_group"], ["hprd_total", "turnover_total_fn"], cfg)
    gov = out.filter(pl.col("value") == "government")
    for r in gov.iter_rows(named=True):
        assert r["null_rate"] == pytest.approx(0.4, abs=0.01), r["feature"]
        assert r["flag"] is True


def test_missingness_small_slice_never_flagged(cfg, gold_frame):
    df = build_slice_columns(gold_frame, cfg).with_columns(
        pl.when(pl.col("ccn") == "100000")
        .then(pl.lit("VT"))
        .otherwise(pl.col(slice_col("state")))
        .alias(slice_col("state")),
        pl.when(pl.col("ccn") == "100000")
        .then(None)
        .otherwise(pl.col("hprd_total"))
        .alias("hprd_total"),
    )
    out = missingness_by_slice(df, ["state"], ["hprd_total"], cfg)
    vt = out.filter(pl.col("value") == "VT").row(0, named=True)
    assert vt["null_rate"] == 1.0 and vt["small_slice"] is True and vt["flag"] is False


def test_small_slice_homes_or_rows(cfg):
    def rep(n_fac, months):
        df = build_slice_columns(make_gold(n_fac=n_fac * 3, months=months), cfg)
        r = representation(df, [SliceDef(("ownership_group",), ("government",))], cfg)
        return r.row(0, named=True)

    small = rep(40, 1)  # 40 homes, 40 rows: enough homes, too few rows
    assert (small["n_facilities"], small["n_rows"], small["small_slice"]) == (40, 40, True)
    ok = rep(30, 24)  # exactly 30 homes and 720 rows: not small
    assert (ok["n_facilities"], ok["n_rows"], ok["small_slice"]) == (30, 720, False)


def test_latest_scope_is_the_last_month(cfg, gold_frame):
    rows = apply_scope(gold_frame, cfg, "latest", "y_g12")
    assert set(rows["t"]) == {gold_frame["t"].max()}


def test_used_counts_only_labeled_rows(cfg):
    df = make_gold(n_fac=10, months=2).with_columns(
        pl.when(pl.col("t") == "2024-02")
        .then(pl.lit("immature"))
        .otherwise(pl.col("status_g12"))
        .alias("status_g12")
    )
    c = status_totals(df, cfg, ["y_g12"])["y_g12"]
    assert c["immature"] == 10
    assert c["used"] == c["positive"] + c["negative"] == 10


def test_boolean_slice_tested_once(cfg):
    df = build_slice_columns(make_gold(rate={"government": 0.6}), cfg)
    ranked = find_problematic(label_rates(df, enumerate_slices(df, cfg), cfg, ["y_g12"]), cfg)
    for name in ("in_hospital", "ccrc"):
        assert ranked.filter(pl.col("slice") == name)["value"].to_list() == ["true"]


def test_risk_tiers_never_problematic(cfg):
    df = make_gold(n_fac=300).with_columns(
        pl.when(pl.col("ccn").str.slice(-1).cast(pl.Int32) < 3)
        .then(pl.lit("SFF"))
        .otherwise(pl.lit("NONE"))
        .alias("sff_status")
    )
    df = df.with_columns(
        pl.when(pl.col("sff_status") == "SFF").then(1).otherwise(0).cast(pl.Int8).alias("y_g12")
    ).with_columns(
        pl.when(pl.col("y_g12") == 1)
        .then(pl.lit("positive"))
        .otherwise(pl.lit("negative"))
        .alias("status_g12")
    )
    df = build_slice_columns(df, cfg)
    ranked = find_problematic(label_rates(df, enumerate_slices(df, cfg), cfg, ["y_g12"]), cfg)
    sff = ranked.filter((pl.col("slice") == "sff_status") & (pl.col("value") == "SFF")).row(
        0, named=True
    )
    assert sff["risk_tier"] is True and sff["adj_p"] < 0.05 and abs(sff["cohens_h"]) > 1
    assert sff["problematic"] is False
    assert not ranked.filter(pl.col("risk_tier"))["problematic"].any()
    assert math.isclose(cohens_h(1.0, 0.0), math.pi)
