"""Evaluation metrics for one set of labels and scores (docs/ML_PIPELINE_SPEC.md §6.1).

Every function takes plain arrays: ``y`` (0/1 labels) and ``p`` (scores; calibrated probabilities where a metric
needs them). The test frame holds one row per facility and test month, so the same functions serve the overall
numbers and the per-month ones. Ported from PR #49 (``exp_common.py``, ``final_model_metrics.py``); the tests
check them against scikit-learn where it has the same metric.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score

from ..training.data import average_precision

Array = npt.NDArray[np.float64]


def _arrays(y: npt.ArrayLike, p: npt.ArrayLike) -> tuple[Array, Array]:
    ya, pa = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if ya.shape != pa.shape or ya.ndim != 1:
        raise ValueError(
            f"y and p must be 1-D arrays of the same length, got {ya.shape} and {pa.shape}"
        )
    if ya.size == 0:
        raise ValueError("no rows")
    if np.isnan(pa).any():
        raise ValueError("scores contain NaN")
    return ya, pa


def pct(frac: float) -> str:
    """0.05 -> "5", 0.025 -> "2.5": the suffix of ``precision_at_<pct>`` and friends."""
    return f"{frac * 100:g}"


def top_k(y: npt.ArrayLike, p: npt.ArrayLike, frac: float) -> dict[str, float]:
    """Precision, recall and lift among the ``frac`` highest-scored rows (at least one row; ties by row order).

    ``k = round(frac x rows)``, so the same fraction flags the same number of rows for every model on a frame.
    """
    ya, pa = _arrays(y, p)
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    k = max(1, int(round(len(ya) * frac)))
    top = np.argsort(-pa, kind="stable")[:k]
    hits = float(ya[top].sum())
    positives, base = float(ya.sum()), float(ya.mean())
    precision = hits / k
    return {
        "k": float(k),
        "precision": precision,
        "recall": hits / positives if positives else float("nan"),
        "lift": precision / base if base else float("nan"),
    }


def ranking_metrics(
    y: npt.ArrayLike, p: npt.ArrayLike, fractions: Sequence[float]
) -> dict[str, float]:
    """PR-AUC, its lift over the base rate, ROC-AUC, and precision, recall and lift at each top fraction."""
    ya, pa = _arrays(y, p)
    base = float(ya.mean())
    both = 0 < base < 1
    pr_auc = average_precision(ya, pa) if base > 0 else float("nan")
    out = {
        "rows": float(len(ya)),
        "positives": float(ya.sum()),
        "base_rate": base,
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / base if base > 0 else float("nan"),
        "roc_auc": float(roc_auc_score(ya, pa)) if both else float("nan"),
    }
    for frac in fractions:
        t = top_k(ya, pa, frac)
        out[f"precision_at_{pct(frac)}"] = t["precision"]
        out[f"recall_at_{pct(frac)}"] = t["recall"]
        out[f"lift_at_{pct(frac)}"] = t["lift"]
    return out


def reliability_table(y: npt.ArrayLike, p: npt.ArrayLike, bins: int) -> list[dict[str, float]]:
    """``bins`` equal-mass bins of the sorted scores (``np.array_split``, ties by row order): per bin its rows, score
    range, mean predicted probability, observed rate and their gap (observed minus predicted)."""
    ya, pa = _arrays(y, p)
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    order = np.argsort(pa, kind="stable")
    rows = []
    for i, idx in enumerate(np.array_split(order, min(bins, len(ya)))):
        predicted, observed = float(pa[idx].mean()), float(ya[idx].mean())
        rows.append(
            {
                "bin": float(i + 1),
                "rows": float(len(idx)),
                "p_min": float(pa[idx].min()),
                "p_max": float(pa[idx].max()),
                "mean_predicted": predicted,
                "observed_rate": observed,
                "gap": observed - predicted,
            }
        )
    return rows


def ece(y: npt.ArrayLike, p: npt.ArrayLike, bins: int) -> float:
    """Expected calibration error: the row-weighted mean |observed - predicted| over equal-mass bins."""
    table = reliability_table(y, p, bins)
    n = sum(b["rows"] for b in table)
    return sum(b["rows"] * abs(b["gap"]) for b in table) / n


def probability_metrics(y: npt.ArrayLike, p: npt.ArrayLike, bins: int) -> dict[str, float]:
    """Brier score, its skill against always predicting the base rate, ECE, the mean prediction vs the rate, and
    the calibration slope and intercept."""
    ya, pa = _arrays(y, p)
    if ((pa < 0) | (pa > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    base = float(ya.mean())
    brier = float(np.mean((pa - ya) ** 2))
    reference = base * (1 - base)  # the Brier score of predicting the base rate for every row
    return {
        "brier": brier,
        "brier_reference": reference,
        "brier_skill": 1 - brier / reference if reference else float("nan"),
        "ece": ece(ya, pa, bins),
        "mean_predicted": float(pa.mean()),
        "observed_rate": base,
        **{f"calibration_{k}": v for k, v in calibration_line(ya, pa).items()},
    }


def calibration_line(y: npt.ArrayLike, p: npt.ArrayLike, clip: float = 1e-6) -> dict[str, float]:
    """Slope and intercept of the logistic recalibration ``y ~ sigmoid(a * logit(p) + b)`` (gate G4).

    Perfect calibration is slope 1, intercept 0; a slope below 1 means the scores are too spread out (too sure).
    Fitted by Newton's method on the two-parameter, unpenalised log-likelihood. Probabilities are clipped to
    ``[clip, 1 - clip]`` before the logit.
    """
    ya, pa = _arrays(y, p)
    if not 0 < ya.mean() < 1:
        return {"slope": float("nan"), "intercept": float("nan")}
    q = np.clip(pa, clip, 1 - clip)
    x = np.column_stack([np.log(q / (1 - q)), np.ones(len(q))])
    beta = np.array([1.0, 0.0])
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(x @ beta)))
        grad = x.T @ (ya - mu)
        hess = (x * (mu * (1 - mu))[:, None]).T @ x
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    return {"slope": float(beta[0]), "intercept": float(beta[1])}
