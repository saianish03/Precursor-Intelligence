"""Population stability index (docs/ML_PIPELINE_SPEC.md §12.1).

PSI = sum over bins of (cur - ref) * ln(cur / ref), on the share of rows per bin. Numeric columns use the
reference's quantile bins (``bins`` of them, fewer when values repeat); categorical and boolean columns use their
levels. Nulls are a bin of their own in both. Empty bins get a small floor so the log stays finite.
"""

from __future__ import annotations

import numpy as np
import polars as pl

FLOOR = 1e-4
NULL = "<null>"


def _shares(labels: np.ndarray, levels: np.ndarray) -> np.ndarray:
    counts = np.array([(labels == lv).sum() for lv in levels], dtype=float)
    return np.clip(counts / max(len(labels), 1), FLOOR, None)


def psi(reference: pl.Series, current: pl.Series, bins: int = 10) -> float:
    """PSI of ``current`` against ``reference``. Raises ValueError on an empty series (nothing to compare)."""
    if reference.len() == 0 or current.len() == 0:
        raise ValueError("PSI needs rows in both the reference and the current month")
    if reference.dtype.is_numeric():
        ref = reference.cast(pl.Float64).to_numpy()
        cur = current.cast(pl.Float64).to_numpy()
        edges = np.unique(
            np.nanquantile(ref, np.linspace(0, 1, bins + 1)[1:-1]) if np.isfinite(ref).any() else []
        )

        def label(x: np.ndarray) -> np.ndarray:
            return np.where(np.isnan(x), -1, np.searchsorted(edges, x, side="right"))

        ref_l, cur_l = label(ref), label(cur)
    else:
        ref_l = reference.cast(pl.String).fill_null(NULL).to_numpy()
        cur_l = current.cast(pl.String).fill_null(NULL).to_numpy()
    levels = np.union1d(np.unique(ref_l), np.unique(cur_l))
    p, q = _shares(ref_l, levels), _shares(cur_l, levels)
    return float(np.sum((q - p) * np.log(q / p)))
