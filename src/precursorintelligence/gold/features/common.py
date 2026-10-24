"""Helpers every feature family uses: look-back windows, day counts, fills and footnote reasons (spec §8)."""

from __future__ import annotations

from functools import cache

import polars as pl

from ...common.config import ConfigError, FeaturesConfig, load_labels_config
from .. import GoldCheckError

CUTOFF = pl.col("cutoff_date")
SEVERITIES = "ABCDEFGHIJKL"  # ordinal severity: A = 1 ... L = 12 (spec §8.1)


@cache
def gplus_min_ordinal() -> int:
    """The ordinal of the least severe G+ letter, from ``configs/labels.yaml: gplus_severities``, the set the
    labels use, so features and labels never disagree on what G+ is. Features test ``max severity >= it``, which
    equals "any G+" only when the set is a letter and everything above it, so any other set is refused."""
    letters = sorted(load_labels_config().gplus_severities)
    start = SEVERITIES.index(letters[0])
    if letters != list(SEVERITIES[start:]):
        raise ConfigError(
            f"labels.yaml gplus_severities {letters} must be one letter and every letter above it"
        )
    return start + 1


def gplus_severities() -> list[str]:
    """The G+ letters (from labels.yaml), e.g. G-L."""
    return list(SEVERITIES[gplus_min_ordinal() - 1 :])


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
