"""Distribution skew per slice (Lecture 5, TFDV-style): within each slice value, how far each feature's
distribution moved between the reference rows (``train``) and the comparison rows (``test``).

Numeric features: Jensen-Shannon distance (base 2, in [0, 1]) on shared bins whose edges come from the
reference, plus one overflow bin on each side for comparison values outside the reference range.
Categorical and boolean features: L-infinity distance, the largest absolute difference in category share.
Without split columns, the earliest third of months is the reference and the latest third the comparison.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from scipy.spatial.distance import jensenshannon

from .config import BiasConfig
from .slices import MISSING, slice_col

TIME_THIRDS = "time_thirds"


def js_distance(p: Sequence[float] | np.ndarray, q: Sequence[float] | np.ndarray) -> float:
    """Jensen-Shannon distance (base 2) between two count or probability vectors of the same length."""
    pa, qa = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if pa.sum() == 0 or qa.sum() == 0:
        return float("nan")
    return float(jensenshannon(pa / pa.sum(), qa / qa.sum(), base=2))


def numeric_js(ref: np.ndarray, cmp: np.ndarray, bins: int) -> float:
    """JS distance of two samples on ``bins`` reference-range bins plus an overflow bin on each side."""
    ref, cmp = ref[np.isfinite(ref)], cmp[np.isfinite(cmp)]
    if ref.size == 0 or cmp.size == 0:
        return float("nan")
    inner = np.histogram_bin_edges(ref, bins=bins)
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    # values equal to the top reference edge belong to the last inner bin, not the overflow
    cmp = np.where(cmp == inner[-1], np.nextafter(inner[-1], -np.inf), cmp)
    ref = np.where(ref == inner[-1], np.nextafter(inner[-1], -np.inf), ref)
    return js_distance(np.histogram(ref, edges)[0], np.histogram(cmp, edges)[0])


def linf_distance(ref: Sequence[Any], cmp: Sequence[Any]) -> float:
    """Largest absolute difference in category frequency (nulls are their own category)."""
    if len(ref) == 0 or len(cmp) == 0:
        return float("nan")
    a, b = Counter(ref), Counter(cmp)
    return float(max(abs(a[k] / len(ref) - b[k] / len(cmp)) for k in set(a) | set(b)))


def split_scopes(df: pl.DataFrame, cfg: BiasConfig) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    """(reference rows, comparison rows, method): the configured splits, or time thirds without them."""
    split = cfg.split_column(cfg.skew.split_label)
    if split in df.columns:
        ref = df.filter(pl.col(split) == cfg.skew.reference_scope)
        cmp = df.filter(pl.col(split) == cfg.skew.comparison_scope)
        return ref, cmp, f"{split}: {cfg.skew.reference_scope} vs {cfg.skew.comparison_scope}"
    months = sorted(df["t"].unique().to_list())
    third = max(len(months) // 3, 1)
    early, late = months[:third], months[-third:]
    ref = df.filter(pl.col("t").is_in(early))
    cmp = df.filter(pl.col("t").is_in(late))
    return ref, cmp, f"{TIME_THIRDS}: {early[0]}..{early[-1]} vs {late[0]}..{late[-1]}"


def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype.is_numeric() and dtype != pl.Boolean


def _small(part: pl.DataFrame, cfg: BiasConfig) -> bool:
    thr = cfg.thresholds
    return part.height < thr.min_slice_rows or part["ccn"].n_unique() < thr.min_slice_facilities


def skew_by_slice(
    df: pl.DataFrame, names: list[str], features: list[str], cfg: BiasConfig
) -> tuple[pl.DataFrame, str]:
    """slice, value, feature, type, distance, metric, flag, n_ref, n_cmp; and the scope method used.

    Slice values that are small in either scope are skipped.
    """
    ref_all, cmp_all, method = split_scopes(df, cfg)
    feats = features[: cfg.skew.max_features] if cfg.skew.max_features else features
    rows = []
    for name in names:
        col = slice_col(name)
        values = sorted(set(ref_all[col].unique().to_list()) & set(cmp_all[col].unique().to_list()))
        for v in values:
            ref, cmp = ref_all.filter(pl.col(col) == v), cmp_all.filter(pl.col(col) == v)
            if _small(ref, cfg) or _small(cmp, cfg):
                continue
            for f in feats:
                dtype = df.schema[f]
                if _is_numeric(dtype):
                    d = numeric_js(
                        ref[f].cast(pl.Float64).to_numpy(),
                        cmp[f].cast(pl.Float64).to_numpy(),
                        cfg.skew.numeric_bins,
                    )
                    kind, metric, limit = "numeric", "js", cfg.thresholds.skew_js
                else:
                    d = linf_distance(
                        ref[f].cast(pl.String).fill_null(MISSING).to_list(),
                        cmp[f].cast(pl.String).fill_null(MISSING).to_list(),
                    )
                    kind, metric, limit = "categorical", "linf", cfg.thresholds.skew_linf
                dist = None if np.isnan(d) else d
                rows.append(
                    {
                        "slice": name,
                        "value": v,
                        "feature": f,
                        "type": kind,
                        "distance": dist,
                        "metric": metric,
                        "flag": bool(dist is not None and dist > limit),
                        "n_ref": ref.height,
                        "n_cmp": cmp.height,
                    }
                )
    schema = {
        "slice": pl.String,
        "value": pl.String,
        "feature": pl.String,
        "type": pl.String,
        "distance": pl.Float64,
        "metric": pl.String,
        "flag": pl.Boolean,
        "n_ref": pl.Int64,
        "n_cmp": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema), method
