"""Mitigation helper: per-row sample weights for the ML phase. The data is never changed.

Weights are fit on, and written for, the training rows only (``split_<weight_label> == "train"``), whatever
the report's scope: a weight that read calibration, purge, test or score rows would leak held-out labels into
training. Every other row gets a null weight.

  inverse_frequency   w = N / (K * n_slice), so every slice value carries the same total weight
  balance_label_rate  Kamiran-Calders reweighing on labeled rows, w = P(slice) * P(y) / P(slice, y), so each
                      slice's weighted label rate equals the overall rate (unlabeled rows get no weight: null)
  none                w = 1

Weights are clipped to ``mitigation.clip`` and renormalised to mean 1 (repeated until both hold). A clip that
binds breaks the equal-total (inverse_frequency) and equal-rate (balance_label_rate) properties for the
clipped slice values: they keep less weight than the formula gives them.
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from .config import BiasConfig
from .slices import slice_col

log = logging.getLogger(__name__)

TRAIN = "train"
CLIP_ROUNDS = 100


def effective_sample_size(weights: pl.Series) -> float:
    """Kish's effective sample size, (sum w)^2 / sum w^2, over the non-null weights."""
    w = weights.drop_nulls().cast(pl.Float64)
    sq = float((w * w).sum())
    return float(w.sum()) ** 2 / sq if sq > 0 else 0.0


def fit_rows(df: pl.DataFrame, cfg: BiasConfig) -> pl.Series:
    """Boolean mask of the rows weights are fit on and written for: ``split_<weight_label> == "train"``."""
    split = cfg.split_column(cfg.mitigation.weight_label)
    if split not in df.columns:
        log.warning("weights: no split column %s, so no training rows; every weight is null", split)
        return pl.Series("_fit", [False] * df.height, dtype=pl.Boolean)
    return df.select((pl.col(split) == TRAIN).fill_null(False).alias("_fit"))["_fit"]


def _renormalise(w: pl.Series, lo: float, hi: float) -> pl.Series:
    """Clip to [lo, hi] and rescale to mean 1, until the mean is 1 within float error."""
    for _ in range(CLIP_ROUNDS):
        mean = w.mean()
        if not mean:
            break
        w = (w / float(mean)).clip(lo, hi)  # type: ignore[arg-type]
        if abs(float(w.mean()) - 1.0) < 1e-12:  # type: ignore[arg-type]
            break
    return w


def sample_weights(df: pl.DataFrame, cfg: BiasConfig, scheme: str | None = None) -> pl.DataFrame:
    """ccn, t, weight, scheme: one row per row of ``df``, in its order; weight is null outside the training
    rows (fit_rows), and nothing outside them is read."""
    m = cfg.mitigation
    scheme = scheme or m.weights
    col = slice_col(m.weight_slice)
    fit = fit_rows(df, cfg)
    train = df.filter(fit)
    base = train.select("ccn", "t", col)
    if scheme == "none":
        w = base.with_columns(pl.lit(1.0).alias("weight"))
    elif scheme == "inverse_frequency":
        k = base[col].n_unique()
        w = base.with_columns(
            (pl.len() / (k * pl.len().over(col))).cast(pl.Float64).alias("weight")
        )
    elif scheme == "balance_label_rate":
        label = m.weight_label
        is_lab = (
            pl.col(cfg.status_column(label)).is_in(cfg.labeled_values) & pl.col(label).is_not_null()
        )
        lab = train.filter(is_lab).select(col, pl.col(label).cast(pl.Int8).alias("_y"))
        n = lab.height
        p_s = lab.group_by(col).agg((pl.len() / n).alias("_ps"))
        p_y = lab.group_by("_y").agg((pl.len() / n).alias("_py"))
        p_sy = lab.group_by(col, "_y").agg((pl.len() / n).alias("_psy"))
        w = (
            train.select(
                "ccn", "t", col, is_lab.alias("_lab"), pl.col(label).cast(pl.Int8).alias("_y")
            )
            .join(p_s, on=col, how="left")
            .join(p_y, on="_y", how="left")
            .join(p_sy, on=[col, "_y"], how="left", maintain_order="left")
            .with_columns(
                pl.when(pl.col("_lab"))
                .then(pl.col("_ps") * pl.col("_py") / pl.col("_psy"))
                .otherwise(None)
                .alias("weight")
            )
        )
    else:
        raise ValueError(f"unknown weight scheme {scheme}")
    lo, hi = m.clip
    fitted = w.select(
        "ccn", "t", _renormalise(w["weight"].cast(pl.Float64), lo, hi).alias("weight")
    )
    return (
        df.select("ccn", "t")
        .join(fitted, on=["ccn", "t"], how="left", maintain_order="left")
        .with_columns(pl.lit(scheme).alias("scheme"))
    )


def ess_report(weights: pl.DataFrame, df: pl.DataFrame, cfg: BiasConfig) -> dict[str, Any]:
    """Effective sample size over the weighted (training) rows, overall and per value of the weight slice."""
    col = slice_col(cfg.mitigation.weight_slice)
    joined = weights.join(df.select("ccn", "t", col), on=["ccn", "t"], how="left").filter(
        pl.col("weight").is_not_null()
    )
    per = {
        str(v): {"n": part.height, "ess": effective_sample_size(part["weight"])}
        for (v,), part in sorted(joined.group_by(col), key=lambda kv: str(kv[0][0]))
    }
    return {
        "scheme": weights["scheme"][0] if weights.height else cfg.mitigation.weights,
        "weight_slice": cfg.mitigation.weight_slice,
        "fit_on": f"{cfg.split_column(cfg.mitigation.weight_label)} == {TRAIN}",
        "n": joined.height,
        "ess": effective_sample_size(joined["weight"]),
        "by_slice": per,
    }
