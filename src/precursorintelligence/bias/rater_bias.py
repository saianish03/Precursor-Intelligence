"""Rater (measurement) bias by state (Lecture 4): labels come from state survey agencies, so a state that
inspects more often can show more citations without its homes being worse. Per state: the label rate, the
inspection-exposure statistics and the state's G+ rate, plus the Spearman correlation across states.

Flagged states (label rate and an exposure both above the configured quantile) are measurement context, a
"possible inspection-intensity effect", not a judgement of facility quality.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import polars as pl
from scipy.stats import spearmanr

from .config import BiasConfig
from .disparity import labeled


def rater_bias_by_state(
    df: pl.DataFrame, cfg: BiasConfig, labels: Sequence[str]
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """(per-state table for every label, summary with Spearman rho per label and exposure)."""
    rb = cfg.rater_bias
    group, exposures = rb.group_column, rb.exposure_columns
    stats = (
        df.group_by(group)
        .agg(
            pl.len().alias("n_rows"),
            *(pl.col(c).mean().alias(f"mean_{c}") for c in exposures),
            *(pl.col(c).median().alias(f"median_{c}") for c in exposures),
            pl.col(rb.rate_column).mean().alias(rb.rate_column),
        )
        .with_columns(pl.col(group).cast(pl.String).fill_null("missing"))
    )
    frames, summary = [], {}
    for label in labels:
        lab = labeled(df, cfg, label)
        rates = (
            lab.group_by(group)
            .agg(pl.len().alias("n"), pl.col(label).cast(pl.Float64).mean().alias("label_rate"))
            .with_columns(pl.col(group).cast(pl.String).fill_null("missing"))
        )
        t = stats.join(rates, on=group, how="left").with_columns(pl.col("n").fill_null(0))
        known = t.filter(pl.col("label_rate").is_not_null())
        flag = pl.lit(False)
        if known.height:
            rate_q = known["label_rate"].quantile(rb.quantile, "linear")
            exp_flag = pl.lit(False)
            for c in exposures:
                q = known[f"mean_{c}"].quantile(rb.quantile, "linear")
                if q is not None:
                    exp_flag = exp_flag | (pl.col(f"mean_{c}") > q)
            if rate_q is not None:
                flag = (pl.col("label_rate") > rate_q) & exp_flag
        t = t.with_columns(flag.fill_null(False).alias("flag"), pl.lit(label).alias("label"))
        rho = {}
        for c in [*exposures, rb.rate_column]:
            col = c if c == rb.rate_column else f"mean_{c}"
            pairs = known.select("label_rate", col).drop_nulls()
            if (
                pairs.height >= 3
                and pairs["label_rate"].n_unique() > 1
                and pairs[col].n_unique() > 1
            ):
                res = spearmanr(pairs["label_rate"].to_numpy(), pairs[col].to_numpy())
                r, p = float(res.statistic), float(res.pvalue)
                rho[c] = {
                    "rho": None if math.isnan(r) else r,
                    "p_value": None if math.isnan(p) else p,
                }
            else:
                rho[c] = {"rho": None, "p_value": None}
        summary[label] = {
            "spearman": rho,
            "flagged_states": sorted(t.filter(pl.col("flag"))[group].to_list()),
            "quantile": rb.quantile,
        }
        frames.append(
            t.select(
                "label",
                pl.col(group).alias("state"),
                "n",
                "label_rate",
                "n_rows",
                *(f"mean_{c}" for c in exposures),
                *(f"median_{c}" for c in exposures),
                rb.rate_column,
                "flag",
            ).sort("state")
        )
    return pl.concat(frames), summary
