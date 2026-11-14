"""Facility block bootstrap (docs/ML_PIPELINE_SPEC.md §6.3): confidence intervals that resample homes, not rows.

A home's rows in different months share most of their 12-month outcome window, so they are not independent.
Each replicate draws as many facilities as there are, with replacement, and keeps all rows of every facility
drawn. On a frame with one row per facility (one test month) this is the ordinary row bootstrap.
Ported from PR #49 (``exp8_all_tables.py: cluster_boot``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Metric = Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], Mapping[str, float]]


def facility_resamples(
    groups: npt.ArrayLike, replicates: int, seed: int
) -> Iterator[npt.NDArray[np.intp]]:
    """``replicates`` arrays of row indices, each the rows of a with-replacement draw of the facilities.

    The same ``groups``, ``replicates`` and ``seed`` always give the same draws, so two models scored on the
    same rows are compared on the same resampled facilities (a paired comparison).
    """
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")
    ga = np.asarray(groups)
    if ga.ndim != 1 or ga.size == 0:
        raise ValueError("groups must be a non-empty 1-D array, one facility id per row")
    uniq, codes = np.unique(ga, return_inverse=True)
    order = np.argsort(codes, kind="stable")  # rows grouped by facility
    starts = np.r_[0, np.cumsum(np.bincount(codes, minlength=len(uniq)))]
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        picked = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([order[starts[g] : starts[g + 1]] for g in picked])


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimates on all rows and the replicate values of every metric, for every score, on shared draws.

    ``replicates`` counts the draws used: a draw whose rows hold one class only is skipped, since PR-AUC and
    ROC-AUC are undefined there.
    """

    # score -> metric -> its value on all rows, and its value on each replicate
    estimates: dict[str, dict[str, float]]
    draws: dict[str, dict[str, npt.NDArray[np.float64]]]
    replicates: int
    level: float

    def ci(self, score: str, metric: str) -> tuple[float, float]:
        """Percentile interval of one metric of one score."""
        return _percentiles(self.draws[score][metric], self.level)

    def difference(self, score: str, reference: str, metric: str) -> dict[str, float]:
        """``metric(score) - metric(reference)`` with its paired interval: both on the same resampled facilities,
        so what the two models share (the same hard homes) cancels out (§6.3, the gates in §9.1)."""
        low, high = _percentiles(
            self.draws[score][metric] - self.draws[reference][metric], self.level
        )
        estimate = self.estimates[score][metric] - self.estimates[reference][metric]
        return {"estimate": estimate, "ci_low": low, "ci_high": high}


def _percentiles(values: npt.NDArray[np.float64], level: float) -> tuple[float, float]:
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    tail = (1 - level) / 2 * 100
    return float(np.percentile(finite, tail)), float(np.percentile(finite, 100 - tail))


def facility_bootstrap(
    y: npt.ArrayLike,
    scores: Mapping[str, npt.ArrayLike],
    groups: npt.ArrayLike,
    metric: Metric,
    *,
    replicates: int,
    seed: int,
    level: float = 0.95,
) -> BootstrapResult:
    """Every score's metrics on all rows and on ``replicates`` facility resamples (the same draws for every
    score). ``metric(y, p)`` returns a dict of named values; all scores share the labels ``y`` and rows."""
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    ya = np.asarray(y, dtype=np.float64)
    arrays = {name: np.asarray(p, dtype=np.float64) for name, p in scores.items()}
    if not arrays:
        raise ValueError("no scores to bootstrap")
    if any(p.shape != ya.shape for p in arrays.values()) or np.shape(groups) != ya.shape:
        raise ValueError("every score and groups must have one value per label")
    estimates = {name: dict(metric(ya, p)) for name, p in arrays.items()}
    values: dict[str, dict[str, list[float]]] = {
        name: {m: [] for m in est} for name, est in estimates.items()
    }
    used = 0
    for idx in facility_resamples(groups, replicates, seed):
        yb = ya[idx]
        if not 0 < yb.mean() < 1:
            continue
        used += 1
        for name, p in arrays.items():
            for m, v in metric(yb, p[idx]).items():
                values[name][m].append(v)
    draws = {
        name: {m: np.asarray(v, dtype=np.float64) for m, v in per.items()}
        for name, per in values.items()
    }
    return BootstrapResult(estimates=estimates, draws=draws, replicates=used, level=level)
