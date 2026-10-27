"""Mitigation helper: per-row sample weights for the ML phase. The data is never changed.

  inverse_frequency   w = N / (K * n_slice), so every slice value carries the same total weight
  balance_label_rate  Kamiran-Calders reweighing on labeled rows, w = P(slice) * P(y) / P(slice, y), so each
                      slice's weighted label rate equals the overall rate (unlabeled rows get no weight: null)
  none                w = 1

Weights are normalised to mean 1, then clipped to ``mitigation.clip``.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from .config import BiasConfig
from .disparity import labeled
from .slices import slice_col


def effective_sample_size(weights: pl.Series) -> float:
    """Kish's effective sample size, (sum w)^2 / sum w^2, over the non-null weights."""
    w = weights.drop_nulls().cast(pl.Float64)
    sq = float((w * w).sum())
    return float(w.sum()) ** 2 / sq if sq > 0 else 0.0


def sample_weights(df: pl.DataFrame, cfg: BiasConfig, scheme: str | None = None) -> pl.DataFrame:
    """ccn, t, weight, scheme, keyed like gold (one row per row of ``df``, in gold order)."""
    m = cfg.mitigation
    scheme = scheme or m.weights
    col = slice_col(m.weight_slice)
    base = df.select("ccn", "t", col)
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
        lab = labeled(df, cfg, label).select(col, pl.col(label).cast(pl.Int8).alias("_y"))
        n = lab.height
        p_s = lab.group_by(col).agg((pl.len() / n).alias("_ps"))
        p_y = lab.group_by("_y").agg((pl.len() / n).alias("_py"))
        p_sy = lab.group_by(col, "_y").agg((pl.len() / n).alias("_psy"))
        w = (
            df.select(
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
    mean = w.select(pl.col("weight").mean()).item()
    scale = 1.0 / float(mean) if mean else 1.0
    return w.select(
        "ccn",
        "t",
        (pl.col("weight") * scale).clip(lo, hi).alias("weight"),
        pl.lit(scheme).alias("scheme"),
    )


def ess_report(weights: pl.DataFrame, df: pl.DataFrame, cfg: BiasConfig) -> dict[str, Any]:
    """Effective sample size overall and per value of the weight slice."""
    col = slice_col(cfg.mitigation.weight_slice)
    joined = weights.with_columns(df[col].alias(col))
    per = {
        str(v): {"n": part.height, "ess": effective_sample_size(part["weight"])}
        for (v,), part in sorted(joined.group_by(col), key=lambda kv: str(kv[0][0]))
    }
    return {
        "scheme": weights["scheme"][0] if weights.height else cfg.mitigation.weights,
        "weight_slice": cfg.mitigation.weight_slice,
        "n": weights.height,
        "ess": effective_sample_size(weights["weight"]),
        "by_slice": per,
    }
