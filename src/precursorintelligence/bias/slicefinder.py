"""SliceFinder-style testing (Lecture 4): which slices' label rates differ from the rest, by enough to matter
(Cohen's h) and beyond chance after Benjamini-Hochberg false-discovery control across all tests of a label.

The test is a z-test on the difference of rates with a cluster-robust (by home) variance
(disparity.clustered_variance), because a home's labels are correlated across months; Fisher's exact test
(on rows) replaces it when any observed cell of the 2x2 table has fewer than 5 rows, and a slice of a single
home is not tested (p = 1, ``too_few_homes``). A slice that is small in
this label's labeled rows (fewer than ``min_slice_facilities`` homes or ``min_slice_rows`` rows) is never
problematic, however large its effect.

A two-valued (boolean) slice is tested on its ``true`` side only: ``false`` against its complement is the same
comparison mirrored. Risk tiers (``slices.risk_tiers``, built from past inspections) are tested and ranked
like any slice but are never problematic: a higher rate there is expected, not bias.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import polars as pl
from scipy.stats import fisher_exact, norm
from statsmodels.stats.multitest import multipletests

from .config import BiasConfig
from .slices import NAME_SEP

MIN_CELL = 5  # below this in any observed 2x2 cell, Fisher's exact test replaces the z-test
# one home cannot carry a test: its residuals sum to 0 around its own rate, so its variance is 0 (p = 1)
MIN_HOMES = 2
BOOLEAN_SLICES = ("in_hospital", "ccrc")
MIRRORED = "false"  # the side of a boolean slice that is not tested


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


def clustered_z_test(diff: float, var_diff: float | None) -> float:
    """Two-sided p-value of a difference of rates with its cluster-robust variance."""
    if var_diff is None or var_diff <= 0:
        return 1.0 if diff == 0 else 0.0
    return float(2 * norm.sf(abs(diff) / math.sqrt(var_diff)))


def bh_adjust(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (statsmodels ``fdr_bh``)."""
    if not pvalues:
        return []
    return [float(p) for p in multipletests(list(pvalues), method="fdr_bh")[1]]


def find_problematic(rates: pl.DataFrame, cfg: BiasConfig) -> pl.DataFrame:
    """Test every (label, slice value) of ``rates`` (label_rates) with labeled rows on both sides, adjust
    p-values per label, and rank them per label: problematic first, then by |h|, then by adj_p
    (``top_k`` keeps the report's head).
    """
    thr = cfg.thresholds
    frames = []
    for _, part in sorted(rates.group_by("label"), key=lambda kv: str(kv[0][0])):
        part = part.with_columns(
            (
                (pl.col("n_homes") < thr.min_slice_facilities) | (pl.col("n") < thr.min_slice_rows)
            ).alias("small_slice"),
            _is_risk_tier(cfg).alias("risk_tier"),
        )
        testable = part.filter(_testable())
        rows = []
        for r in testable.iter_rows(named=True):
            x1, n1, x2, n2 = r["positives"], r["n"], r["complement_positives"], r["complement_n"]
            if r["n_homes"] < MIN_HOMES:
                p, test = 1.0, "too_few_homes"
            elif min(x1, n1 - x1, x2, n2 - x2) < MIN_CELL:
                p, test = two_proportion_test(x1, n1, x2, n2)
            else:
                p, test = clustered_z_test(x1 / n1 - x2 / n2, r["var_diff"]), "z_cluster"
            h = cohens_h(x1 / n1, x2 / n2)
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
                and not r["risk_tier"]
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
                "n_homes",
                "rate",
                "complement_rate",
                "cohens_h",
                "p_value",
                "adj_p",
                "test_used",
                "small_slice",
                "risk_tier",
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
                "n_homes": pl.Int64,
                "rate": pl.Float64,
                "complement_rate": pl.Float64,
                "cohens_h": pl.Float64,
                "p_value": pl.Float64,
                "adj_p": pl.Float64,
                "test_used": pl.String,
                "small_slice": pl.Boolean,
                "risk_tier": pl.Boolean,
                "problematic": pl.Boolean,
                "rank": pl.Int64,
            }
        )
    return pl.concat(frames)


def _is_risk_tier(cfg: BiasConfig) -> pl.Expr:
    """True for a slice (single or crossing) that involves a risk tier."""
    parts = pl.col("slice").str.split(NAME_SEP)
    return parts.list.eval(pl.element().is_in(cfg.slices.risk_tiers)).list.any()


def _testable() -> pl.Expr:
    """Labeled rows on both sides, and not the mirrored side of a boolean slice."""
    mirrored = pl.col("slice").is_in(BOOLEAN_SLICES) & (pl.col("value") == MIRRORED)
    return (pl.col("n") > 0) & (pl.col("complement_n") > 0) & ~mirrored


def top_k(problematic: pl.DataFrame, k: int) -> pl.DataFrame:
    """The first ``k`` ranked rows of every label."""
    return problematic.filter(pl.col("rank") <= k)


def n_tests(rates: pl.DataFrame) -> int:
    """How many (label, slice value) pairs find_problematic tests."""
    return rates.filter(_testable()).height
