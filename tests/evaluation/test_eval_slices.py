"""evaluation/slices.py: per-slice calibration and flag metrics (docs/ML_PIPELINE_SPEC.md §7)."""

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score

from precursorintelligence.bias.config import load_bias_config
from precursorintelligence.common.config import load_ml_config
from precursorintelligence.evaluation.slices import (
    MISSING,
    bias_check_oof,
    calibration_failures,
    gated,
    gated_slices,
    slice_table,
)
from precursorintelligence.training.select import default_bias_check


def homes(n_homes: int = 400, months: int = 3, seed: int = 0, shift: float = 0.0):
    """Rows of ``n_homes`` facilities over ``months`` months. For-profit homes' true risk is ``shift`` above what
    the score says; everyone else is scored at their true risk."""
    rng = np.random.default_rng(seed)
    ccn = np.array([f"{i:06d}" for i in range(n_homes)])
    owner = rng.choice(["for_profit", "non_profit", "government"], n_homes, p=[0.6, 0.3, 0.1])
    beds = rng.choice([30.0, 80.0, 120.0, 250.0], n_homes)
    beds[:5] = np.nan
    base = rng.beta(2, 5, n_homes)
    rows = pl.DataFrame(
        {
            "ccn": np.repeat(ccn, months),
            "t": np.tile([f"2023-0{m + 1}" for m in range(months)], n_homes),
            "ownership_group": np.repeat(owner, months),
            "certified_beds": np.repeat(beds, months),
            "state": np.repeat(rng.choice(["CA", "TX"], n_homes), months),
        }
    )
    p = np.repeat(base, months)
    true = np.clip(p + np.where(rows["ownership_group"].to_numpy() == "for_profit", shift, 0), 0, 1)
    y = (rng.random(len(p)) < true).astype(int)
    return rows, y, p


CFG = load_bias_config()


def table(rows, y, p, **kw):
    kw.setdefault("slices", ["ownership_group", "bed_band"])
    return slice_table(rows, y, p, bias_cfg=CFG, **kw)


def row(t: pl.DataFrame, slice_: str, value: str) -> dict:
    return t.filter((pl.col("slice") == slice_) & (pl.col("value") == value)).row(0, named=True)


def test_counts_and_o_minus_e_per_value():
    rows, y, p = homes()
    t = table(rows, y, p)
    mask = rows["ownership_group"].to_numpy() == "non_profit"
    r = row(t, "ownership_group", "non_profit")
    assert r["rows"] == mask.sum()
    assert (
        r["facilities"] == rows.filter(pl.col("ownership_group") == "non_profit")["ccn"].n_unique()
    )
    assert r["positives"] == y[mask].sum()
    assert r["base_rate"] == pytest.approx(y[mask].mean())
    assert r["mean_p"] == pytest.approx(p[mask].mean())
    assert r["oe"] == pytest.approx(y[mask].mean() - p[mask].mean())
    assert r["oe_ci_low"] < r["oe"] < r["oe_ci_high"]


def test_an_underestimated_group_has_an_interval_above_zero():
    rows, y, p = homes(n_homes=3000, shift=0.1)
    t = table(rows, y, p)
    assert row(t, "ownership_group", "for_profit")["oe_ci_low"] > 0.05
    other = row(t, "ownership_group", "non_profit")
    assert other["oe_ci_low"] < 0 < other["oe_ci_high"]


def test_repeating_a_homes_months_does_not_shrink_the_interval():
    # three identical months per home carry no more information than one: the clustered SE must not drop
    rows1, y1, p1 = homes(n_homes=500, months=1, seed=3)
    rows3 = pl.concat([rows1] * 3)
    y3, p3 = np.tile(y1, 3), np.tile(p1, 3)
    se1 = row(table(rows1, y1, p1), "ownership_group", "for_profit")["oe_se"]
    se3 = row(table(rows3, y3, p3), "ownership_group", "for_profit")["oe_se"]
    assert se3 == pytest.approx(se1)


def test_flag_metrics_use_one_global_cut_off():
    rows, y, p = homes(seed=4)
    t = table(rows, y, p, flag_frac=0.2)
    k = round(len(p) * 0.2)
    flag = np.zeros(len(p), bool)
    flag[np.argsort(-p, kind="stable")[:k]] = True
    mask = rows["ownership_group"].to_numpy() == "for_profit"
    r = row(t, "ownership_group", "for_profit")
    yy, ff = y[mask] == 1, flag[mask]
    assert r["flag_share"] == pytest.approx(ff.mean())
    assert r["precision"] == pytest.approx((ff & yy).sum() / ff.sum())
    assert r["recall"] == pytest.approx((ff & yy).sum() / yy.sum())
    assert r["fnr"] == pytest.approx(1 - r["recall"])
    assert r["fpr"] == pytest.approx((ff & ~yy).sum() / (~yy).sum())


def test_pr_auc_within_the_slice_needs_enough_positives():
    rows, y, p = homes(seed=5)
    t = table(rows, y, p, min_pr_auc_positives=30)
    mask = rows["ownership_group"].to_numpy() == "for_profit"
    assert row(t, "ownership_group", "for_profit")["pr_auc"] == pytest.approx(
        average_precision_score(y[mask], p[mask])
    )
    t = table(rows, y, p, min_pr_auc_positives=10_000)
    assert t["pr_auc"].is_nan().all()


def test_small_slices_are_marked_unreliable_and_nulls_are_missing():
    rows, y, p = homes(seed=6)
    t = table(rows, y, p, min_facilities=50)
    missing = row(t, "bed_band", MISSING)  # the five homes without a bed count
    assert missing["facilities"] == 5 and missing["reliable"] is False
    assert row(t, "ownership_group", "for_profit")["reliable"] is True


def test_rows_y_and_p_must_line_up():
    rows, y, p = homes()
    with pytest.raises(ValueError, match="same"):
        table(rows, y[:-1], p)


def gold_rows(n_homes: int, shift: float, seed: int = 0):
    """``homes`` with every gate slice's gold column and the label, as the eval window's rows arrive."""
    rows, y, p = homes(n_homes=n_homes, months=2, seed=seed, shift=shift)
    rng = np.random.default_rng(seed + 1)
    per_home = rows.height // n_homes
    chain = rng.choice(["C1", "C2", None], n_homes)
    rows = rows.with_columns(
        pl.Series("t", np.tile(["2023-07", "2023-08"], n_homes)),
        pl.Series("chain_id", np.repeat(chain, per_home)),
        pl.Series("provider_in_hospital", np.repeat(rng.random(n_homes) < 0.1, per_home)),
        pl.Series("y_g12", y.astype(np.int8)),
    )
    return rows, p


def test_bias_check_passes_a_calibrated_score():
    rows, p = gold_rows(4000, shift=0.0)
    out = bias_check_oof(rows, p, load_ml_config())
    assert out.passed, out.detail
    assert "0 outside" in out.detail["bias_check"] and "failing" not in out.detail


def test_bias_check_fails_a_group_the_score_underestimates():
    rows, p = gold_rows(4000, shift=0.12)
    out = bias_check_oof(rows, p, load_ml_config())
    assert not out.passed
    assert "ownership_group=for_profit" in out.detail["failing"]


def test_state_is_reported_but_not_gated():
    cfg = load_ml_config()
    assert "state" in cfg.bias.gate_slices and "state" not in gated_slices(cfg)


def test_small_or_missing_values_are_not_gated():
    cfg = load_ml_config()
    t = pl.DataFrame(
        {
            "slice": ["ownership_group", "ownership_group", "bed_band", "state"],
            "value": ["for_profit", "government", MISSING, "CA"],
            "facilities": [5000, 10, 5000, 5000],
            "positives": [900, 5, 900, 900],
            "oe": [0.2, 0.2, 0.2, 0.2],
            "oe_ci_low": [0.15, 0.15, 0.15, 0.15],
            "oe_ci_high": [0.25, 0.25, 0.25, 0.25],
        }
    )
    assert gated(t, cfg)["value"].to_list() == ["for_profit"]
    assert calibration_failures(t, cfg)["value"].to_list() == ["for_profit"]


def test_w1_selection_finds_this_check():
    assert default_bias_check() is bias_check_oof


def test_slice_comparisons_see_a_group_get_worse():
    from precursorintelligence.evaluation.slices import slice_comparisons

    rows, y, p = homes(n_homes=3000, seed=8)
    worse = p + np.where(rows["ownership_group"].to_numpy() == "for_profit", -0.1, 0.0)
    values = pl.DataFrame({"slice": ["ownership_group"] * 2, "value": ["for_profit", "non_profit"]})
    out = {
        c.value: c
        for c in slice_comparisons(
            rows, y, worse, p, values=values, bias_cfg=CFG, replicates=100, seed=0
        )
    }
    assert out["for_profit"].abs_oe_diff.ci_low > 0.05  # its calibration got clearly worse
    same = out["non_profit"]
    assert same.abs_oe_diff.estimate == pytest.approx(0) and same.pr_auc_diff.estimate == 0


def test_no_values_no_comparisons():
    from precursorintelligence.evaluation.slices import slice_comparisons

    rows, y, p = homes()
    empty = pl.DataFrame({"slice": [], "value": []}, schema={"slice": pl.Utf8, "value": pl.Utf8})
    assert slice_comparisons(rows, y, p, p, values=empty, bias_cfg=CFG, replicates=5, seed=0) == []
