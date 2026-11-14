"""Sanity checks of the evaluation itself (docs/ML_PIPELINE_SPEC.md §6.4), feeding gate G2.

- **Shuffled labels:** refit LightGBM with the eval window's labels shuffled within each month and score the test
  frame. With no signal left, test ROC-AUC must be near 0.5 (``evaluation.shuffled_label.roc_range``); anything
  else means the evaluation sees something it should not (a join, an index, a split bug). It does not catch
  feature leakage: a leaking feature has no link to shuffled labels either.
- **SHAP additivity:** on a sample of test rows, the explanation's base plus the sum of its contributions is the
  calibrated log-odds (§8.1), so the reasons a home is shown add up to its score.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.metrics import roc_auc_score

from ..common.config import FeatureSpec, MlConfig
from ..training.data import to_model_frame
from ..training.ensemble import PrecursorModel
from ..training.members import LgbmMember, categorical_levels

SHAP_ROWS = 1000
SHAP_TOLERANCE = 1e-6


def shuffle_within_groups(
    y: npt.ArrayLike, groups: npt.ArrayLike, seed: int
) -> npt.NDArray[np.float64]:
    """``y`` permuted inside each group (month): every month keeps its base rate, the link to the rows is gone."""
    ya, ga = np.asarray(y, dtype=np.float64), np.asarray(groups)
    if ya.shape != ga.shape:
        raise ValueError("y and groups must have the same shape")
    rng = np.random.default_rng(seed)
    out = ya.copy()
    for g in np.unique(ga):
        idx = np.flatnonzero(ga == g)
        out[idx] = ya[rng.permutation(idx)]
    return out


def shuffled_label_roc(
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: Sequence[FeatureSpec],
    cfg: MlConfig,
    *,
    seed: int = 0,
) -> float:
    """Test ROC-AUC of a LightGBM (``members.lgbm`` with ``shuffled_label.n_estimators`` trees) fit on ``train``
    with its labels shuffled within month."""
    label = f"y_{cfg.label}"
    params = cfg.members.lgbm.model_copy(
        update={"n_estimators": cfg.evaluation.shuffled_label.n_estimators}
    )
    x = to_model_frame(train, features)
    y = shuffle_within_groups(train[label].to_numpy(), train["t"].to_numpy(), seed)
    member = LgbmMember.fit(x, y, features, params, levels=categorical_levels(x, features))
    score = member.logit(to_model_frame(test, features))
    return float(roc_auc_score(test[label].to_numpy(), score))


@dataclass(frozen=True)
class Additivity:
    rows: int
    max_abs_error: float
    tolerance: float

    @property
    def holds(self) -> bool:
        return self.max_abs_error <= self.tolerance


def shap_additivity(
    model: PrecursorModel,
    df: pl.DataFrame,
    *,
    rows: int = SHAP_ROWS,
    seed: int = 0,
    tolerance: float = SHAP_TOLERANCE,
) -> Additivity:
    """The largest |base + sum of contributions - calibrated log-odds| over ``rows`` sampled rows of ``df``."""
    if df.is_empty():
        raise ValueError("no rows to check SHAP additivity on")
    sample = df.sample(n=min(rows, df.height), seed=seed)
    base, contrib = model.explain(sample)
    logit = model.predict_frame(sample)["logit"].to_numpy()
    err = np.abs(np.asarray(base) + contrib.to_numpy().sum(axis=1) - logit)
    return Additivity(rows=sample.height, max_abs_error=float(err.max()), tolerance=tolerance)
