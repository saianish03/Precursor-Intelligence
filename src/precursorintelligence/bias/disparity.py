"""Label-rate disparity (group fairness at the data level): per slice value, the label rate, its Wilson
interval, the rate of every other labeled row (the complement), their ratio and difference.

Grouped rates come from Fairlearn's ``MetricFrame``; the interval and the complement are computed here.
Only rows whose label status is in ``labeled_values`` count; immature and censored rows are counted apart.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from fairlearn.metrics import MetricFrame
from scipy.stats import norm

from .config import BiasConfig
from .slices import SliceDef, group_columns, slice_col, slice_frame, with_slice_value

STATUS_COUNTS = ("positive", "negative", "immature", "censored")


def wilson_ci(positives: int, n: int, level: float) -> tuple[float | None, float | None]:
    """Wilson score interval for positives / n; (None, None) when n = 0."""
    if n <= 0:
        return None, None
    z = float(norm.ppf(0.5 + level / 2))
    p = positives / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def labeled(df: pl.DataFrame, cfg: BiasConfig, label: str) -> pl.DataFrame:
    """Rows of ``df`` whose ``status_<label>`` is labeled (positive / negative) and whose label is set."""
    return df.filter(
        pl.col(cfg.status_column(label)).is_in(cfg.labeled_values) & pl.col(label).is_not_null()
    )


def _rate(y_true: Any, y_pred: Any) -> float:
    return float(np.mean(y_true))


def _count(y_true: Any, y_pred: Any) -> int:
    return len(y_true)


def grouped_rates(lab: pl.DataFrame, label: str, cols: tuple[str, ...]) -> pl.DataFrame:
    """Fairlearn MetricFrame over ``lab`` grouped by the slice columns: slice, value, n, rate."""
    schema = {"slice": pl.String, "value": pl.String, "n": pl.Int64, "rate": pl.Float64}
    if lab.height == 0:
        return pl.DataFrame(schema=schema)
    names = [slice_col(c) for c in cols]
    sens = lab.select(names).to_pandas()
    y = lab[label].cast(pl.Int64).to_numpy()
    mf = MetricFrame(
        metrics={"rate": _rate, "count": _count}, y_true=y, y_pred=y, sensitive_features=sens
    )
    # crossings come back as the full product of levels; combinations with no rows are NaN
    by = mf.by_group.reset_index().dropna(subset=["count"])
    g = pl.DataFrame(
        {
            **{n: by[n].astype(str).tolist() for n in names},
            "n": by["count"].astype("int64").tolist(),
            "rate": by["rate"].astype("float64").tolist(),
        }
    )
    return with_slice_value(g, cols).select(list(schema))


def label_rates(
    df: pl.DataFrame, slices: Sequence[SliceDef], cfg: BiasConfig, labels: Sequence[str]
) -> pl.DataFrame:
    """One row per (label, slice value); see the module docstring. Values with no labeled rows get n = 0."""
    thr = cfg.thresholds
    frames = []
    for label in labels:
        lab = labeled(df, cfg, label)
        total_n = lab.height
        total_pos = int(lab[label].cast(pl.Int64).sum() or 0)
        status = cfg.status_column(label)
        rates = []
        mix = []
        for cols in group_columns(slices):
            rates.append(grouped_rates(lab, label, cols))
            m = df.group_by([slice_col(c) for c in cols]).agg(
                (pl.col(status) == s).sum().alias(f"n_{s}") for s in STATUS_COUNTS
            )
            mix.append(
                with_slice_value(m, cols).select(
                    "slice", "value", *(f"n_{s}" for s in STATUS_COUNTS)
                )
            )
        out = slice_frame(slices).join(pl.concat(rates), on=["slice", "value"], how="left")
        out = out.join(pl.concat(mix), on=["slice", "value"], how="left")
        out = out.with_columns(
            pl.col("n").fill_null(0),
            *(pl.col(f"n_{s}").fill_null(0).cast(pl.Int64) for s in STATUS_COUNTS),
        ).with_columns((pl.col("rate") * pl.col("n")).round(0).cast(pl.Int64).alias("positives"))
        out = out.with_columns(
            pl.col("positives").fill_null(0),
            pl.when(pl.col("n") > 0).then(pl.col("rate")).otherwise(None).alias("rate"),
            (total_n - pl.col("n")).alias("complement_n"),
        ).with_columns((total_pos - pl.col("positives")).alias("complement_positives"))
        out = out.with_columns(
            pl.when(pl.col("complement_n") > 0)
            .then(pl.col("complement_positives") / pl.col("complement_n"))
            .otherwise(None)
            .alias("complement_rate")
        )
        out = out.with_columns(
            pl.when(pl.col("complement_rate") > 0)
            .then(pl.col("rate") / pl.col("complement_rate"))
            .otherwise(None)
            .alias("ratio"),
            (pl.col("rate") - pl.col("complement_rate")).alias("diff"),
        ).with_columns(
            ((pl.col("ratio") < thr.ratio_low) | (pl.col("ratio") > thr.ratio_high))
            .fill_null(False)
            .alias("flag_ratio")
        )
        cis = [wilson_ci(p, n, thr.ci_level) for p, n in out.select("positives", "n").rows()]
        out = out.with_columns(
            pl.Series("ci_low", [c[0] for c in cis], dtype=pl.Float64),
            pl.Series("ci_high", [c[1] for c in cis], dtype=pl.Float64),
            pl.lit(label).alias("label"),
        )
        frames.append(
            out.sort("order").select(
                "label",
                "slice",
                "value",
                "n",
                "positives",
                "rate",
                "ci_low",
                "ci_high",
                "complement_n",
                "complement_positives",
                "complement_rate",
                "ratio",
                "diff",
                "flag_ratio",
                "n_immature",
                "n_censored",
            )
        )
    return pl.concat(frames)


def status_totals(
    df: pl.DataFrame, cfg: BiasConfig, labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """label -> rows per status (positive / negative / immature / censored), for the log and summary."""
    out = {}
    for label in labels:
        counts = dict(df[cfg.status_column(label)].value_counts().iter_rows())
        out[label] = {s: int(counts.get(s, 0)) for s in STATUS_COUNTS}
        out[label]["used"] = labeled(df, cfg, label).height
    return out
