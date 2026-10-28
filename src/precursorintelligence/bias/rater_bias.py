"""Rater (measurement) bias by state (Lecture 4): labels come from state survey agencies, so a state that
inspects more often can show more citations without its homes being worse. Per state: the label rate, the
inspection-exposure statistics and the state's past G+ rate, plus the Spearman correlation across states.

Every state statistic is computed on the same rows, the label's labeled rows. Exposure is inspection
frequency and backlog (``n_std_36m``, ``days_since_last_std``, ``std_overdue``), not complaint counts: CMS
lists a complaint survey only when it led to a citation, so that count is conditioned on the outcome. The
correlation with the state's past G+ rate (``rate_column``) measures persistence, not rater bias. States with
fewer than ``min_state_facilities`` homes are reported but take no part in the quantiles, the correlation or
the flags. When ``severity_columns`` and ``citation_total_column`` are set, ``gplus_share`` (the state's G+
citations over all its citations) measures how severely the state cites; it is reported, never flagged.

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


def _spearman(known: pl.DataFrame, col: str) -> dict[str, float | None]:
    pairs = known.select("label_rate", col).drop_nulls()
    if pairs.height < 3 or pairs["label_rate"].n_unique() < 2 or pairs[col].n_unique() < 2:
        return {"rho": None, "p_value": None}
    res = spearmanr(pairs["label_rate"].to_numpy(), pairs[col].to_numpy())
    r, p = float(res.statistic), float(res.pvalue)
    return {"rho": None if math.isnan(r) else r, "p_value": None if math.isnan(p) else p}


def rater_bias_by_state(
    df: pl.DataFrame, cfg: BiasConfig, labels: Sequence[str]
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """(per-state table for every label, summary with Spearman rho per label and exposure)."""
    rb = cfg.rater_bias
    group, exposures = rb.group_column, rb.exposure_columns
    severity = bool(
        rb.severity_columns
        and rb.citation_total_column
        and all(c in df.columns for c in [*rb.severity_columns, rb.citation_total_column])
    )
    share = (
        [
            (
                pl.sum_horizontal(pl.col(c).cast(pl.Float64) for c in rb.severity_columns).sum()
                / pl.col(rb.citation_total_column).cast(pl.Float64).sum()
            ).alias("gplus_share")
        ]
        if severity
        else []
    )
    frames, summary = [], {}
    for label in labels:
        lab = labeled(df, cfg, label)
        t = (
            lab.group_by(group)
            .agg(
                pl.len().alias("n"),
                pl.col("ccn").n_unique().alias("n_facilities"),
                pl.col(label).cast(pl.Float64).mean().alias("label_rate"),
                *(pl.col(c).cast(pl.Float64).mean().alias(f"mean_{c}") for c in exposures),
                *(pl.col(c).cast(pl.Float64).median().alias(f"median_{c}") for c in exposures),
                pl.col(rb.rate_column).cast(pl.Float64).mean().alias(rb.rate_column),
                *share,
            )
            .with_columns(
                pl.col(group).cast(pl.String).fill_null("missing"),
                (pl.col("n_facilities") >= rb.min_state_facilities).alias("eligible"),
            )
        )
        known = t.filter(pl.col("eligible") & pl.col("label_rate").is_not_null())
        flag = pl.lit(False)
        if known.height:
            rate_q = known["label_rate"].quantile(rb.quantile, "linear")
            exp_flag = pl.lit(False)
            for c in exposures:
                q = known[f"mean_{c}"].quantile(rb.quantile, "linear")
                if q is not None:
                    exp_flag = exp_flag | (pl.col(f"mean_{c}") > q)
            if rate_q is not None:
                flag = pl.col("eligible") & (pl.col("label_rate") > rate_q) & exp_flag
        t = t.with_columns(flag.fill_null(False).alias("flag"), pl.lit(label).alias("label"))
        rho = {c: _spearman(known, f"mean_{c}") for c in exposures}
        summary[label] = {
            "spearman": rho,
            "persistence": {rb.rate_column: _spearman(known, rb.rate_column)},
            "severity": {"gplus_share": _spearman(known, "gplus_share")} if severity else {},
            "flagged_states": sorted(t.filter(pl.col("flag"))[group].to_list()),
            "quantile": rb.quantile,
            "states_ranked": known.height,
            "min_state_facilities": rb.min_state_facilities,
        }
        frames.append(
            t.select(
                "label",
                pl.col(group).alias("state"),
                "n",
                "n_facilities",
                "label_rate",
                *(f"mean_{c}" for c in exposures),
                *(f"median_{c}" for c in exposures),
                rb.rate_column,
                *(["gplus_share"] if severity else []),
                "eligible",
                "flag",
            ).sort("state")
        )
    return pl.concat(frames), summary
