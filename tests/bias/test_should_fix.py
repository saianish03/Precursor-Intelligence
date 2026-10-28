"""Should-fix items from the PR #26 review: skew against the population's own drift (S1), rater-bias
exposures (S2), --gold provenance (S3) and censoring by slice (S5)."""

import hashlib
import json

import polars as pl
import pytest
from bias_fakes import make_gold

from precursorintelligence.bias import BiasInputError
from precursorintelligence.bias.disparity import censoring_by_slice, label_rates, status_totals
from precursorintelligence.bias.io import load_gold_path
from precursorintelligence.bias.skew import skew_by_slice
from precursorintelligence.bias.slices import build_slice_columns, enumerate_slices
from precursorintelligence.gold import GoldCheckError


def _loose(cfg):
    return cfg.model_copy(
        update={"thresholds": cfg.thresholds.model_copy(update={"min_slice_rows": 100})}
    )


# --------------------------------------------------------------------------- S1


def test_population_drift_is_not_slice_skew(cfg):
    """Every home's feature moves between train and test: large distance, flagged for no slice."""
    cfg = _loose(cfg)
    df = make_gold().with_columns(
        pl.when(pl.col("split_g12") == "test")
        .then(pl.col("hprd_total") + 0.5)
        .otherwise(pl.col("hprd_total"))
        .alias("hprd_total")
    )
    out, _ = skew_by_slice(build_slice_columns(df, cfg), ["ownership_group"], ["hprd_total"], cfg)
    assert (out["distance"] > cfg.thresholds.skew_js).all()
    assert (out["population_distance"] > cfg.thresholds.skew_js).all()
    assert not out["flag"].any()


def test_slice_specific_drift_is_flagged(cfg):
    cfg = _loose(cfg)
    df = make_gold().with_columns(
        pl.when((pl.col("split_g12") == "test") & (pl.col("ownership_group") == "government"))
        .then(pl.col("hprd_total") + 0.5)
        .otherwise(pl.col("hprd_total"))
        .alias("hprd_total")
    )
    out, _ = skew_by_slice(build_slice_columns(df, cfg), ["ownership_group"], ["hprd_total"], cfg)
    flagged = dict(out.select("value", "flag").iter_rows())
    assert flagged == {"for_profit": False, "government": True, "non_profit": False}
    gov = out.filter(pl.col("value") == "government").row(0, named=True)
    assert gov["distance"] > gov["null_q"]


def test_geo_slices_skip_state_level_features(cfg):
    cfg = _loose(cfg)
    df = build_slice_columns(make_gold(), cfg)
    out, _ = skew_by_slice(df, ["state", "ownership_group"], ["state_gplus_rate_12m"], cfg)
    assert set(out["slice"]) == {"ownership_group"}


def test_skew_null_is_deterministic(cfg):
    cfg = _loose(cfg)
    df = build_slice_columns(make_gold(), cfg)
    a, _ = skew_by_slice(df, ["ownership_group"], ["days_since_last_std"], cfg)
    b, _ = skew_by_slice(df, ["ownership_group"], ["days_since_last_std"], cfg)
    assert a.equals(b)


# --------------------------------------------------------------------------- S2


def test_rater_exposures_are_inspection_frequency(cfg):
    assert "n_complaint_surveys_cited_12m" not in cfg.rater_bias.exposure_columns
    assert {"n_std_36m", "days_since_last_std", "std_overdue"} <= set(
        cfg.rater_bias.exposure_columns
    )


# --------------------------------------------------------------------------- S3


def _gold_with_pointer(tmp_path, df):
    """gold/<table>/t=*/part-0.parquet plus a gold/_current.json listing their sha256."""
    gold = tmp_path / "gold"
    shas = {}
    for (t,), part in sorted(df.group_by("t"), key=lambda kv: str(kv[0][0])):
        d = gold / "facility_month_dataset" / f"t={t}"
        d.mkdir(parents=True)
        f = d / "part-0.parquet"
        part.drop("t").write_parquet(f)
        shas[f"facility_month_dataset/t={t}"] = hashlib.sha256(f.read_bytes()).hexdigest()
    current = {"state": "current", "run_id": "r1", "v": "2024-12", "verified": True, "sha256": shas}
    (gold / "_current.json").write_text(json.dumps(current), encoding="utf-8")
    return gold


def test_gold_path_with_pointer_is_verified(cfg, tmp_path):
    df = make_gold(n_fac=30, months=3)
    gold = _gold_with_pointer(tmp_path, df)
    data = load_gold_path(gold, cfg)
    assert data.df.height == df.height
    assert (data.pointer["run_id"], data.pointer["partitions"]) == ("r1", 3)
    # a partition changed after the pointer was written is refused
    f = gold / "facility_month_dataset" / "t=2024-02" / "part-0.parquet"
    pl.read_parquet(f).head(5).write_parquet(f)
    with pytest.raises(GoldCheckError, match="does not match"):
        load_gold_path(gold, cfg)


def test_gold_path_ignores_unlisted_partitions(cfg, tmp_path):
    df = make_gold(n_fac=30, months=3)
    gold = _gold_with_pointer(tmp_path, df)
    stray = gold / "facility_month_dataset" / "t=2023-12"
    stray.mkdir()
    df.filter(pl.col("t") == "2024-01").drop("t").write_parquet(stray / "part-0.parquet")
    assert set(load_gold_path(gold, cfg).df["t"]) == {"2024-01", "2024-02", "2024-03"}


def test_gold_path_rejects_repeated_keys(cfg, tmp_path):
    df = make_gold(n_fac=10, months=1)
    f = tmp_path / "one.parquet"
    pl.concat([df, df.head(2)]).write_parquet(f)
    with pytest.raises(BiasInputError, match="share a ccn/t key"):
        load_gold_path(f, cfg)


# --------------------------------------------------------------------------- S5


def test_censoring_by_slice(cfg):
    """Government loses 20% of its matured rows to closures, the others none."""
    df = make_gold(n_fac=300, months=12).with_row_index("i")
    closed = (pl.col("ownership_group") == "government") & (pl.col("i") % 5 == 0)
    df = df.with_columns(
        pl.when(closed)
        .then(pl.lit("censored"))
        .otherwise(pl.col("status_g12"))
        .alias("status_g12"),
        pl.when(closed).then(None).otherwise(pl.col("y_g12")).alias("y_g12"),
    ).drop("i")
    df = build_slice_columns(df, cfg)
    rates = label_rates(df, enumerate_slices(df, cfg), cfg, ["y_g12"])
    out = censoring_by_slice(rates, status_totals(df, cfg, ["y_g12"]), cfg)
    gov = out.filter((pl.col("slice") == "ownership_group") & (pl.col("value") == "government"))
    r = gov.row(0, named=True)
    assert r["n_matured"] == 1200 and r["censored_share"] == pytest.approx(0.2)
    assert r["overall_share"] == pytest.approx(240 / 3600)
    assert r["flag"] is True
    fp = out.filter((pl.col("slice") == "ownership_group") & (pl.col("value") == "for_profit"))
    assert fp.row(0, named=True)["censored_share"] == 0 and fp.row(0, named=True)["flag"] is False
