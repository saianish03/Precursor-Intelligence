"""Platt calibration on out-of-fold log-odds (docs/ML_PIPELINE_SPEC.md §5, ML-24).

``fit_platt`` fits ``p = sigmoid(a * logit + b)`` on the OOF rows of the window's newest ``fit_recent_months``
months. A fit with ``a <= 0`` would reverse or flatten the ranking, so it falls back to the identity and is flagged.
Isotonic is computed only as a diagnostic (``diagnostics``); on its own fitting rows it always wins by construction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .contracts import CalibrationDiagnostics, Candidate, PlattCalibration


def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))), dtype=float)


def recent_months(months: Sequence[str], n: int) -> list[str]:
    """The newest ``n`` months with rows (CMS gap months don't count)."""
    return sorted(set(months))[-n:]


def fit_platt(logit: np.ndarray, y: np.ndarray, months: Sequence[str]) -> PlattCalibration:
    """Unpenalised logistic regression of ``y`` on ``logit`` (two parameters). ``months`` are the fit rows' months,
    recorded in the result."""
    logit = np.asarray(logit, dtype=float)
    y = np.asarray(y, dtype=float)
    if logit.size == 0 or len(np.unique(y)) < 2:
        return PlattCalibration(
            a=1.0, b=0.0, identity=True, fit_rows=int(logit.size), fit_months=list(months)
        )
    lr = LogisticRegression(C=np.inf, max_iter=1000, tol=1e-10)
    lr.fit(logit.reshape(-1, 1), y)
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return PlattCalibration(
            a=1.0,
            b=0.0,
            identity=True,
            fit_rows=int(logit.size),
            fit_months=list(months),
            fitted_a=a,
            fitted_b=b,
        )
    return PlattCalibration(
        a=a, b=b, identity=False, fit_rows=int(logit.size), fit_months=list(months)
    )


def apply_platt(cal: PlattCalibration, logit: np.ndarray) -> np.ndarray:
    """Calibrated log-odds."""
    return cal.a * np.asarray(logit, dtype=float) + cal.b


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over ``bins`` equal-mass bins (rows sorted by p): sum of |mean y - mean p| weighted
    by the bin's share of rows."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    total = 0.0
    for idx in np.array_split(order, min(bins, n)):
        if idx.size:
            total += idx.size / n * abs(float(y[idx].mean()) - float(p[idx].mean()))
    return total


def diagnostics(
    candidate: Candidate,
    rows: str,
    logit_raw: np.ndarray,
    y: np.ndarray,
    cal: PlattCalibration,
    *,
    isotonic_fit: tuple[np.ndarray, np.ndarray] | None = None,
    bins: int = 10,
) -> CalibrationDiagnostics:
    """Brier and ECE of the raw, Platt and isotonic probabilities on ``rows``. Isotonic is fitted on
    ``isotonic_fit`` (the Platt fit rows) and applied here; a diagnostic only."""
    p_raw = sigmoid(logit_raw)
    p_platt = sigmoid(apply_platt(cal, logit_raw))
    fx, fy = isotonic_fit if isotonic_fit is not None else (logit_raw, y)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    if len(np.unique(fy)) >= 2:
        iso.fit(fx, fy)
        p_iso = np.asarray(iso.predict(logit_raw), dtype=float)
    else:
        p_iso = p_raw
    return CalibrationDiagnostics.model_validate(
        {
            "candidate": candidate,
            "rows": rows,
            "brier_raw": brier(y, p_raw),
            "brier_platt": brier(y, p_platt),
            "brier_isotonic": brier(y, p_iso),
            "ece_raw": ece(y, p_raw, bins),
            "ece_platt": ece(y, p_platt, bins),
            "ece_isotonic": ece(y, p_iso, bins),
        }
    )
