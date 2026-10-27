"""SliceFinder-style testing (Lecture 4): which slices' label rates differ from the rest, by enough to matter
(Cohen's h) and beyond chance after Benjamini-Hochberg false-discovery control across all tests of a label.

The test is a pooled two-proportion z-test, or Fisher's exact test when any cell of the 2x2 table has fewer
than 5 rows. A small slice (representation) is never problematic, however large its effect.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import polars as pl
from scipy.stats import fisher_exact, norm
from statsmodels.stats.multitest import multipletests

from .config import BiasConfig

MIN_CELL = 5  # below this in any 2x2 cell, Fisher's exact test replaces the z-test


def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def two_proportion_test(x1: int, n1: int, x2: int, n2: int) -> tuple[float, str]:
    """Two-sided p-value of x1/n1 vs x2/n2, and the test used (``z`` or ``fisher``)."""
    cells = (x1, n1 - x1, x2, n2 - x2)
    if min(cells) < MIN_CELL:
        _, p = fisher_exact([[x1, n1 - x1], [x2, n2 - x2]], alternative="two-sided")
        return float(p), "fisher"
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0, "z"
    z = (x1 / n1 - x2 / n2) / se
    return float(2 * norm.sf(abs(z))), "z"


def bh_adjust(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (statsmodels ``fdr_bh``)."""
    if not pvalues:
        return []
    return [float(p) for p in multipletests(list(pvalues), method="fdr_bh")[1]]


def find_problematic(rates: pl.DataFrame, summary: pl.DataFrame, cfg: BiasConfig) -> pl.DataFrame:
    """Test every (label, slice value) of ``rates`` (label_rates) with labeled rows on both sides, adjust
    p-values per label, and rank them per label: problematic first, then by |h|, then by adj_p
    (``top_k`` keeps the report's head).

    ``summary`` (representation) supplies ``small_slice``.
    """
    thr = cfg.thresholds
    small = summary.select("slice", "value", "small_slice")
    frames = []
    for _, part in sorted(rates.group_by("label"), key=lambda kv: str(kv[0][0])):
        part = part.join(small, on=["slice", "value"], how="left").with_columns(
            pl.col("small_slice").fill_null(True)
        )
        testable = part.filter((pl.col("n") > 0) & (pl.col("complement_n") > 0))
        rows = []
        for r in testable.iter_rows(named=True):
            p, test = two_proportion_test(
                r["positives"], r["n"], r["complement_positives"], r["complement_n"]
            )
            h = cohens_h(r["positives"] / r["n"], r["complement_positives"] / r["complement_n"])
            rows.append({**r, "cohens_h": h, "p_value": p, "test_used": test})
        if not rows:
            continue
        adj = bh_adjust([r["p_value"] for r in rows])
        for r, a in zip(rows, adj, strict=True):
            r["adj_p"] = a
            r["problematic"] = bool(
                a < thr.fdr_alpha
                and abs(r["cohens_h"]) >= thr.effect_size_T
                and not r["small_slice"]
            )
        tested = pl.DataFrame(rows)
        ranked = (
            tested.with_columns(pl.col("cohens_h").abs().alias("_abs_h"))
            .sort(
                ["problematic", "_abs_h", "adj_p", "slice", "value"],
                descending=[True, True, False, False, False],
            )
            .with_row_index("rank", offset=1)
        )
        frames.append(
            ranked.select(
                "label",
                "slice",
                "value",
                "n",
                "rate",
                "complement_rate",
                "cohens_h",
                "p_value",
                "adj_p",
                "test_used",
                "small_slice",
                "problematic",
                pl.col("rank").cast(pl.Int64),
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "label": pl.String,
                "slice": pl.String,
                "value": pl.String,
                "n": pl.Int64,
                "rate": pl.Float64,
                "complement_rate": pl.Float64,
                "cohens_h": pl.Float64,
                "p_value": pl.Float64,
                "adj_p": pl.Float64,
                "test_used": pl.String,
                "small_slice": pl.Boolean,
                "problematic": pl.Boolean,
                "rank": pl.Int64,
            }
        )
    return pl.concat(frames)


def top_k(problematic: pl.DataFrame, k: int) -> pl.DataFrame:
    """The first ``k`` ranked rows of every label."""
    return problematic.filter(pl.col("rank") <= k)


def n_tests(rates: pl.DataFrame) -> int:
    """How many (label, slice value) pairs find_problematic tests."""
    return rates.filter((pl.col("n") > 0) & (pl.col("complement_n") > 0)).height
