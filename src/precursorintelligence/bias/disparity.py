"""Label-rate disparity (group fairness at the data level): per slice value, the label rate, its Wilson
interval, the rate of every other labeled row (the complement), their ratio and difference.

Grouped rates come from Fairlearn's ``MetricFrame``; the interval and the complement are computed here.
Only rows whose label status is in ``labeled_values`` count; immature and censored rows are counted apart.

Inference is by home, not by home-month: a home is labeled in many months and consecutive 12-month windows
overlap, so its labels are strongly correlated. ``clustered_variance`` gives cluster-robust (by ``ccn``)
variances of a slice's rate and of its difference from the complement. The Wilson interval is sized on the
effective n they imply, and slicefinder's z-test uses the variance of the difference.
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


def wilson_ci(positives: float, n: float, level: float) -> tuple[float | None, float | None]:
    """Wilson score interval for positives / n (n may be an effective, non-integer n); (None, None) at n = 0."""
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


def clustered_variance(lab: pl.DataFrame, label: str, cols: tuple[str, ...]) -> pl.DataFrame:
    """Cluster-robust (by ``ccn``) variances per slice value of ``lab`` (labeled rows): slice, value,
    n_homes, var_rate (of the slice's rate) and var_diff (of the slice's rate minus the complement's).

    With e = y - rate of the row's side (slice or complement), home c contributes
    u_c = sum_slice(e) / n1 - sum_complement(e) / n2 to the difference, and var_diff = G/(G-1) sum_c u_c^2 over
    all G homes; a home can sit on both sides (its slice value changed between months). var_rate is the same
    on the slice alone, over its k homes. Either is null with fewer than two homes.
    """
    schema = {
        "slice": pl.String,
        "value": pl.String,
        "n_homes": pl.Int64,
        "var_rate": pl.Float64,
        "var_diff": pl.Float64,
    }
    if lab.height == 0:
        return pl.DataFrame(schema=schema)
    y = pl.col(label).cast(pl.Float64)
    homes = lab.group_by("ccn").agg(pl.len().cast(pl.Float64).alias("_N"), y.sum().alias("_Y"))
    g_all = homes.height
    n_tot, y_tot = float(homes["_N"].sum()), float(homes["_Y"].sum())
    # sum_c (Y_c - N_c p2)^2 over every home, expanded so it is one sum per value of p2
    s_yy, s_yn, s_nn = homes.select(
        (pl.col("_Y") ** 2).sum().alias("yy"),
        (pl.col("_Y") * pl.col("_N")).sum().alias("yn"),
        (pl.col("_N") ** 2).sum().alias("nn"),
    ).row(0)
    cells = with_slice_value(
        lab.group_by(["ccn", *(slice_col(c) for c in cols)]).agg(
            pl.len().cast(pl.Float64).alias("_n"), y.sum().alias("_y")
        ),
        cols,
    ).select("ccn", "slice", "value", "_n", "_y")
    sides = (
        cells.group_by("slice", "value")
        .agg(
            pl.col("_n").sum().alias("_n1"),
            pl.col("_y").sum().alias("_y1"),
            pl.len().alias("n_homes"),
        )
        .with_columns(
            (pl.col("_y1") / pl.col("_n1")).alias("_p1"),
            (n_tot - pl.col("_n1")).alias("_n2"),
        )
        .with_columns(
            pl.when(pl.col("_n2") > 0)
            .then((y_tot - pl.col("_y1")) / pl.col("_n2"))
            .otherwise(None)
            .alias("_p2")
        )
    )
    e1 = pl.col("_y") - pl.col("_n") * pl.col("_p1")  # the home's residuals in the slice
    e2 = (pl.col("_Y") - pl.col("_y")) - (pl.col("_N") - pl.col("_n")) * pl.col("_p2")
    out_of = (pl.col("_Y") - pl.col("_N") * pl.col("_p2")) / pl.col(
        "_n2"
    )  # u_c were it all outside
    per = (
        cells.join(homes, on="ccn", how="left")
        .join(sides, on=["slice", "value"], how="left")
        .group_by("slice", "value")
        .agg(
            (e1**2).sum().alias("_e1"),
            ((e1 / pl.col("_n1") - e2 / pl.col("_n2")) ** 2).sum().alias("_u_in"),
            (out_of**2).sum().alias("_a_in"),
        )
    )
    p2, n1, n2, k = pl.col("_p2"), pl.col("_n1"), pl.col("_n2"), pl.col("n_homes")
    all_out = (s_yy - 2 * p2 * s_yn + p2 * p2 * s_nn) / (n2 * n2)
    return (
        sides.join(per, on=["slice", "value"], how="left")
        .with_columns(
            pl.when(k > 1)
            .then(pl.col("_e1") / (n1 * n1) * k / (k - 1))
            .otherwise(None)
            .alias("var_rate"),
            pl.when((n2 > 0) & (g_all > 1))
            .then(
                (pl.col("_u_in") + all_out - pl.col("_a_in")).clip(lower_bound=0.0)
                * g_all
                / (g_all - 1)
            )
            .otherwise(None)
            .alias("var_diff"),
        )
        .select("slice", "value", k.cast(pl.Int64), "var_rate", "var_diff")
    )


def effective_n(rate: float | None, n: int, n_homes: int, var_rate: float | None) -> float:
    """n / design effect, so that rate (1 - rate) / n_eff = var_rate; at most n. Falls back to the home
    count when the variance is degenerate (rate 0 or 1, or a single home)."""
    if n <= 0:
        return 0.0
    if rate is None or var_rate is None or var_rate <= 0 or not 0 < rate < 1:
        return float(min(n, max(n_homes, 1)))
    return float(min(n, rate * (1 - rate) / var_rate))


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
        clustered = []
        for cols in group_columns(slices):
            rates.append(grouped_rates(lab, label, cols))
            clustered.append(clustered_variance(lab, label, cols))
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
        out = out.join(pl.concat(clustered), on=["slice", "value"], how="left")
        out = out.with_columns(
            pl.col("n").fill_null(0),
            pl.col("n_homes").fill_null(0),
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
            (
                (pl.col("ratio") < thr.ratio_low)
                | (pl.col("ratio") > thr.ratio_high)
                # complement at 0 and the slice above it: the ratio is infinite, the largest gap there is
                | ((pl.col("complement_rate") == 0) & (pl.col("rate") > 0))
            )
            .fill_null(False)
            .alias("flag_ratio")
        )
        n_eff = [
            effective_n(r, n, k, v)
            for r, n, k, v in out.select("rate", "n", "n_homes", "var_rate").rows()
        ]
        cis = [
            wilson_ci((r or 0.0) * ne, ne, thr.ci_level)
            for r, ne in zip(out["rate"].to_list(), n_eff, strict=True)
        ]
        out = out.with_columns(
            pl.Series("n_eff", n_eff, dtype=pl.Float64),
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
                "n_homes",
                "n_eff",
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
                "var_diff",
                "n_immature",
                "n_censored",
            )
        )
    return pl.concat(frames)


def censoring_by_slice(
    rates: pl.DataFrame, totals: dict[str, dict[str, int]], cfg: BiasConfig
) -> pl.DataFrame:
    """Per label and slice value, the share of matured rows (labeled + censored) that are censored (the home
    closed before its window ended) against the label's overall share. Rates above describe surviving homes,
    so a slice that loses many homes this way is flagged (gap >= ``censoring_gap``, not small)."""
    thr = cfg.thresholds
    overall = {
        lb: (c["censored"] / (c["used"] + c["censored"])) if c["used"] + c["censored"] else None
        for lb, c in totals.items()
    }
    return (
        rates.with_columns((pl.col("n") + pl.col("n_censored")).alias("n_matured"))
        .with_columns(
            pl.when(pl.col("n_matured") > 0)
            .then(pl.col("n_censored") / pl.col("n_matured"))
            .otherwise(None)
            .alias("censored_share"),
            pl.col("label")
            .replace_strict(overall, default=None, return_dtype=pl.Float64)
            .alias("overall_share"),
            (
                (pl.col("n_homes") < thr.min_slice_facilities)
                | (pl.col("n_matured") < thr.min_slice_rows)
            ).alias("small_slice"),
        )
        .with_columns((pl.col("censored_share") - pl.col("overall_share")).alias("gap"))
        .with_columns(
            ((pl.col("gap") >= thr.censoring_gap) & ~pl.col("small_slice"))
            .fill_null(False)
            .alias("flag")
        )
        .select(
            "label",
            "slice",
            "value",
            "n_matured",
            "n_censored",
            "censored_share",
            "overall_share",
            "gap",
            "small_slice",
            "flag",
        )
    )


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
