"""Helpers every feature family uses: look-back windows, day counts, fills and footnote reasons (spec §8)."""

from __future__ import annotations

import polars as pl

from ...common.config import FeaturesConfig
from .. import GoldCheckError

CUTOFF = pl.col("cutoff_date")


def within(date_col: str, months: int) -> pl.Expr:
    """``cutoff − months < date <= cutoff``: the last ``months`` months, never after the cutoff."""
    d = pl.col(date_col)
    return (d > CUTOFF.dt.offset_by(f"-{months}mo")) & (d <= CUTOFF)


def days_since(date_expr: pl.Expr) -> pl.Expr:
    """Days from ``date_expr`` to the cutoff, as a float; null when the date is null."""
    return (CUTOFF - date_expr).dt.total_days().cast(pl.Float64)


def attach(rows: pl.DataFrame, per_ccn: pl.DataFrame) -> pl.DataFrame:
    """The spine rows of one t, left-joined to one row per ccn (fails if ``per_ccn`` repeats a ccn)."""
    return rows.join(per_ccn, on="ccn", how="left", validate="1:1", maintain_order="left")


def fill_zero(*cols: str) -> list[pl.Expr]:
    """Counts and sums found nowhere are 0: none found in the published data (spec §8, §10)."""
    return [pl.col(c).fill_null(0) for c in cols]


def footnote_reason(col: str, cfg: FeaturesConfig) -> pl.Expr:
    """A CMS footnote code as its missing reason; no footnote is ``none`` (SILVER_CONTRACT §13.2)."""
    reason = pl.col(col).replace_strict(cfg.footnote_reasons, default=None, return_dtype=pl.String)
    return pl.when(pl.col(col).is_null()).then(pl.lit("none")).otherwise(reason).alias(col)


def check_footnotes(df: pl.DataFrame, cols: list[str], cfg: FeaturesConfig, where: str) -> None:
    """Stop on a footnote code features.yaml does not map, so a new CMS code is mapped on purpose."""
    for col in cols:
        unknown = set(df[col].drop_nulls().unique()) - set(cfg.footnote_reasons)
        if unknown:
            raise GoldCheckError(
                f"{where}.{col}: footnote codes {sorted(unknown)} are not in configs/features.yaml"
            )
