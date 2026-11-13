"""Model performance by facility slice (docs/ML_PIPELINE_SPEC.md §7): is the score as accurate for every group?

Slices are the data-level bias report's (``configs/bias_slices.yaml``, built with ``bias/slices.py``), so a slice
means the same thing in both reports. The criterion is calibration within the group: real risk differs between
groups (for-profit homes are cited more often), so equal flag rates would be wrong, but the probability should be
right in every group. Per slice value:
- O-E, the observed minus the expected (mean predicted) rate, with a 95 % interval that is cluster-robust by
  facility (a home's months share their outcome windows, so they are not independent rows);
- PR-AUC within the slice (when it has enough positives);
- at one global cut-off (the top ``flag_frac`` of all scores): the share flagged, precision, recall, FPR and FNR,
  from Fairlearn's ``MetricFrame``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from scipy.stats import norm
from sklearn.metrics import precision_score, recall_score

from ..bias.config import BiasConfig, SlicesCfg, load_bias_config
from ..bias.slices import MISSING, build_slice_columns, slice_col
from ..common.config import FeatureRegistry, MlConfig, load_feature_registry, project_root
from ..training.data import average_precision
from ..training.select import BiasOutcome

GEOGRAPHIC = (
    "state",
    "cms_region",
)  # G7(b) skips them: state inspection practice, reported per state


def slice_config(cfg: MlConfig) -> BiasConfig:
    """``ml.yaml: bias.slices_config`` (relative to the repo root unless absolute)."""
    path = Path(cfg.bias.slices_config)
    return load_bias_config(path if path.is_absolute() else project_root() / path)


def _pr_auc(min_positives: int) -> Any:
    def pr_auc(y_true: Any, y_score: Any) -> float:
        y = np.asarray(y_true, dtype=np.float64)
        if y.sum() < min_positives or y.sum() == len(y):
            return float("nan")
        return average_precision(y, np.asarray(y_score, dtype=np.float64))

    return pr_auc


def _precision(y_true: Any, y_pred: Any) -> float:
    return float(precision_score(y_true, y_pred, zero_division=np.nan))


def _recall(y_true: Any, y_pred: Any) -> float:
    return float(recall_score(y_true, y_pred, zero_division=np.nan))


def _oe(frame: pl.DataFrame, col: str, z: float) -> pl.DataFrame:
    """Per value of ``col``: O-E = mean(y - p), and its facility-clustered standard error and interval.

    With r = y - p and d the value's mean r, home c contributes u_c = sum over its rows in the value of (r - d);
    var(d) = G / (G - 1) * sum_c u_c^2 / n^2 over the value's G homes and n rows (CR1). Null with one home.
    """
    r = pl.col("_y") - pl.col("_p")
    per_home = frame.group_by(col, "ccn").agg(r.sum().alias("_r"), pl.len().alias("_n"))
    means = frame.group_by(col).agg(r.mean().alias("oe"), pl.len().alias("_rows"))
    return (
        per_home.join(means, on=col)
        .group_by(col)
        .agg(
            pl.first("oe"),
            pl.first("_rows"),
            ((pl.col("_r") - pl.col("_n") * pl.col("oe")) ** 2).sum().alias("_s"),
            pl.len().alias("_g"),
        )
        .with_columns(
            pl.when(pl.col("_g") > 1)
            .then((pl.col("_s") * pl.col("_g") / (pl.col("_g") - 1)).sqrt() / pl.col("_rows"))
            .otherwise(None)
            .alias("oe_se")
        )
        .with_columns(
            (pl.col("oe") - z * pl.col("oe_se")).alias("oe_ci_low"),
            (pl.col("oe") + z * pl.col("oe_se")).alias("oe_ci_high"),
        )
        .select(col, "oe", "oe_se", "oe_ci_low", "oe_ci_high")
    )


def slice_table(
    rows: pl.DataFrame,
    y: npt.ArrayLike,
    p: npt.ArrayLike,
    *,
    slices: Sequence[str],
    bias_cfg: BiasConfig,
    registry: FeatureRegistry | None = None,
    flag_frac: float = 0.10,
    level: float = 0.95,
    min_pr_auc_positives: int = 30,
    min_facilities: int = 30,
) -> pl.DataFrame:
    """One row per slice value of each slice in ``slices``: rows, facilities, positives, base rate, mean p, O-E
    with its interval, PR-AUC, and the flag metrics at the global top-``flag_frac`` cut-off. ``reliable`` is
    False below ``min_facilities`` facilities (reported, marked). ``rows`` are the scored rows with their gold
    columns (``ccn``, ``t`` and each slice's source column), in the order of ``y`` and ``p``."""
    ya, pa = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if not (len(ya) == len(pa) == rows.height) or rows.height == 0:
        raise ValueError("rows, y and p must have the same, non-zero length")
    if not 0 < flag_frac <= 1:
        raise ValueError(f"flag_frac must be in (0, 1], got {flag_frac}")
    only = bias_cfg.model_copy(update={"slices": SlicesCfg(single=list(slices))})
    k = max(1, int(round(len(pa) * flag_frac)))
    flag = np.zeros(len(pa), dtype=np.int64)
    flag[np.argsort(-pa, kind="stable")[:k]] = 1
    frame = build_slice_columns(rows, only, registry).with_columns(
        pl.Series("_y", ya), pl.Series("_p", pa), pl.Series("_flag", flag)
    )
    z = float(norm.ppf(0.5 + level / 2))
    yi = ya.astype(np.int64)
    out = []
    for name in slices:
        col = slice_col(name)
        values = frame[col].to_numpy()
        counts = frame.group_by(col).agg(
            pl.len().alias("rows"),
            pl.col("ccn").n_unique().alias("facilities"),
            pl.col("_y").sum().alias("positives"),
            pl.col("_y").mean().alias("base_rate"),
            pl.col("_p").mean().alias("mean_p"),
        )
        scored = MetricFrame(
            metrics={"pr_auc": _pr_auc(min_pr_auc_positives)},
            y_true=yi,
            y_pred=pa,
            sensitive_features=values,
        ).by_group
        flagged = MetricFrame(
            metrics={
                "n": count,
                "flag_share": selection_rate,
                "precision": _precision,
                "recall": _recall,
                "fpr": false_positive_rate,
                "fnr": false_negative_rate,
            },
            y_true=yi,
            y_pred=flag,
            sensitive_features=values,
        ).by_group
        fair = pl.DataFrame(
            {
                col: [str(v) for v in flagged.index],
                "pr_auc": scored.loc[flagged.index, "pr_auc"].astype(float).to_numpy(),
                **{
                    m: flagged[m].astype(float).to_numpy()
                    for m in ("flag_share", "precision", "recall", "fpr", "fnr")
                },
            }
        )
        out.append(
            counts.join(_oe(frame, col, z), on=col)
            .join(fair, on=col)
            .rename({col: "value"})
            .with_columns(pl.lit(name).alias("slice"))
        )
    return (
        pl.concat(out)
        .with_columns(
            pl.col("positives").cast(pl.Int64),
            (pl.col("facilities") >= min_facilities).alias("reliable"),
        )
        .select(
            "slice",
            "value",
            "rows",
            "facilities",
            "positives",
            "base_rate",
            "mean_p",
            "oe",
            "oe_se",
            "oe_ci_low",
            "oe_ci_high",
            "pr_auc",
            "flag_share",
            "precision",
            "recall",
            "fpr",
            "fnr",
            "reliable",
        )
        .sort("slice", "value")
    )


def gated_slices(cfg: MlConfig) -> list[str]:
    """The slices G7(b) checks: ``bias.gate_slices`` without the geographic ones."""
    return [s for s in cfg.bias.gate_slices if s not in GEOGRAPHIC]


def gated(table: pl.DataFrame, cfg: MlConfig) -> pl.DataFrame:
    """The slice values G7 judges: a slice of ``gated_slices`` with at least ``gate_min_facilities`` facilities
    and ``gate_min_positives`` positives. The ``missing`` value is not a group and is never gated."""
    b = cfg.bias
    return table.filter(
        pl.col("slice").is_in(gated_slices(cfg))
        & (pl.col("value") != MISSING)
        & (pl.col("facilities") >= b.gate_min_facilities)
        & (pl.col("positives") >= b.gate_min_positives)
    )


def calibration_failures(table: pl.DataFrame, cfg: MlConfig) -> pl.DataFrame:
    """G7(b): the gated slice values whose O-E interval lies entirely outside ``±bias.oe_abs_max`` (the score is
    clearly miscalibrated for that group)."""
    limit = cfg.bias.oe_abs_max
    return gated(table, cfg).filter((pl.col("oe_ci_low") > limit) | (pl.col("oe_ci_high") < -limit))


def describe(failures: pl.DataFrame) -> str:
    return "; ".join(
        f"{r['slice']}={r['value']}: O-E {r['oe']:+.3f} [{r['oe_ci_low']:+.3f}, {r['oe_ci_high']:+.3f}]"
        for r in failures.iter_rows(named=True)
    )


def bias_check_oof(rows: pl.DataFrame, p: np.ndarray, cfg: MlConfig) -> BiasOutcome:
    """G7(b) on a candidate's out-of-fold predictions, for selection (§4.3): fails when any gated slice is
    clearly miscalibrated. ``rows`` are the eval window's rows with their gold columns and ``y_<label>``, in
    the order of ``p`` (the calibrated OOF probability). Called by ``training.select.default_bias_check``."""
    slices = gated_slices(cfg)
    if not slices:
        return BiasOutcome(passed=True, detail={"bias_check": "no non-geographic gate slices"})
    table = slice_table(
        rows,
        rows[f"y_{cfg.label}"].to_numpy(),
        p,
        slices=slices,
        bias_cfg=slice_config(cfg),
        registry=load_feature_registry(),
    )
    failures = calibration_failures(table, cfg)
    checked = gated(table, cfg).height
    detail: dict[str, Any] = {
        "bias_check": f"{checked} slice values checked, {failures.height} outside ±{cfg.bias.oe_abs_max}",
    }
    if failures.height:
        detail["failing"] = describe(failures)
    return BiasOutcome(passed=failures.height == 0, detail=detail)
