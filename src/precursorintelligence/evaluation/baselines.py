"""Baseline scores the model is compared with (docs/ML_PIPELINE_SPEC.md §6.2): simple rules from gold columns.

Each takes a frame's rows and returns one score per row, higher = riskier, for the same metrics as the model.
- ``prior_gplus_rule``: 1 if the home had a G-L citation at its last standard survey or any G+ survey in the last
  36 months, else 0. Gate G3 requires the selected model to beat it.
- ``severity_24m``: the citation severity points of the last 24 months, as a ranking score.
- ``star_rating``: the overall star rating, fewer stars = riskier. A benchmark only: ``overall_rating`` is never
  a model feature (data spec §8). Homes without a rating get the frame's mean rating, i.e. no information.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import polars as pl

from ..training import MlInputError

Baseline = Callable[[pl.DataFrame], npt.NDArray[np.float64]]


def _need(df: pl.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise MlInputError(f"baseline columns missing from the frame: {missing}")


def prior_gplus_rule(df: pl.DataFrame) -> npt.NDArray[np.float64]:
    _need(df, "gplus_last_std", "n_gplus_surveys_36m")
    hit = pl.col("gplus_last_std").fill_null(False) | (
        pl.col("n_gplus_surveys_36m").fill_null(0) > 0
    )
    return df.select(hit.cast(pl.Float64))[:, 0].to_numpy()


def severity_24m(df: pl.DataFrame) -> npt.NDArray[np.float64]:
    _need(df, "sev_points_24m")
    return df.select(pl.col("sev_points_24m").cast(pl.Float64).fill_null(0.0))[:, 0].to_numpy()


def star_rating(df: pl.DataFrame) -> npt.NDArray[np.float64]:
    _need(df, "overall_rating")
    stars = pl.col("overall_rating").cast(pl.Float64)
    if df["overall_rating"].null_count() == df.height:
        return np.zeros(df.height)
    return df.select(-stars.fill_null(stars.mean()))[:, 0].to_numpy()


BASELINES: dict[str, Baseline] = {
    "prior_gplus_rule": prior_gplus_rule,
    "severity_24m": severity_24m,
    "star_rating": star_rating,
}


def baseline_scores(df: pl.DataFrame) -> dict[str, npt.NDArray[np.float64]]:
    """Every baseline's score for the rows of ``df``, by name."""
    return {name: fn(df) for name, fn in BASELINES.items()}
