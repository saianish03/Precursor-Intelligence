"""Population stability index (docs/ML_PIPELINE_SPEC.md §12.1).

PSI = sum over bins of (cur - ref) * ln(cur / ref), on the share of rows per bin. Numeric columns use the
reference's quantile bins (``bins`` of them, fewer when values repeat), except group-level features (below) and a
column with one value; categorical and boolean columns use their levels. Nulls are a bin of their own in both.
Empty bins get a small floor so the log stays finite.

A group-level feature carries one value per state or per month for every home (``state_gplus_rate_12m``): a few
dozen non-integer values that all move a little every month. Its quantile edges sit on big states' values, so a move
of a few thousandths empties the bins those states hold alone (PSI 0.96 on the real 2026-10 month, #73 B2). It gets
equal-width bins over the reference's range instead. Integer counts and ratings repeat the same values every month
and keep their quantile bins.
"""

from __future__ import annotations

import numpy as np
import polars as pl

FLOOR = 1e-4
NULL = "<null>"
GROUP_LEVEL_VALUES_PER_BIN = (
    10  # at most bins x this many distinct non-integer values: a group-level feature
)


def _shares(labels: np.ndarray, levels: np.ndarray) -> np.ndarray:
    counts = np.array([(labels == lv).sum() for lv in levels], dtype=float)
    return np.clip(counts / max(len(labels), 1), FLOOR, None)


def _edges(ref: np.ndarray, bins: int) -> np.ndarray:
    finite = ref[np.isfinite(ref)]
    values = np.unique(finite)
    if values.size <= 1:
        # below, at and above the one value, so a column that moves as a whole still drifts
        return np.array([values[0], np.nextafter(values[0], np.inf)]) if values.size else values
    if values.size <= GROUP_LEVEL_VALUES_PER_BIN * bins and np.any(values != np.round(values)):
        return np.linspace(values[0], values[-1], bins + 1)[1:-1]
    return np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1]))


def psi(reference: pl.Series, current: pl.Series, bins: int = 10) -> float:
    """PSI of ``current`` against ``reference``. Raises ValueError on an empty series (nothing to compare)."""
    if reference.len() == 0 or current.len() == 0:
        raise ValueError("PSI needs rows in both the reference and the current month")
    if reference.dtype.is_numeric():
        ref = reference.cast(pl.Float64).to_numpy()
        cur = current.cast(pl.Float64).to_numpy()
        edges = _edges(ref, bins)

        def label(x: np.ndarray) -> np.ndarray:
            return np.where(np.isnan(x), -1, np.searchsorted(edges, x, side="right"))

        ref_l, cur_l = label(ref), label(cur)
    else:
        ref_l = reference.cast(pl.String).fill_null(NULL).to_numpy()
        cur_l = current.cast(pl.String).fill_null(NULL).to_numpy()
    levels = np.union1d(np.unique(ref_l), np.unique(cur_l))
    p, q = _shares(ref_l, levels), _shares(cur_l, levels)
    return float(np.sum((q - p) * np.log(q / p)))
