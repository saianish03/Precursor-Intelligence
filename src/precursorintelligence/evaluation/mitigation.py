"""The bias-mitigation trade-off (docs/ML_PIPELINE_SPEC.md §7): what reweighting would buy, measured on the eval
window's out-of-fold predictions, never on test.

The one documented option is reweighting with ``bias/mitigation.py: sample_weights`` (``configs/bias_slices.yaml:
mitigation``, by default inverse frequency on ``ownership_group``, clipped 0.2-5), fit on the eval frame's own rows.
The selected candidate is refit fold by fold with those weights, exactly as ``train`` builds its OOF predictions
(same folds, members, ensemble weights, and an unweighted Platt fit on the newest months), and compared with
``train``'s unweighted OOF predictions: overall PR-AUC, Brier and ECE against the calibration gaps of the gated
slices. Adopting it is a recipe change (``ml.yaml: bias.mitigation: reweight``) that goes through the gates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

from ..bias.config import BiasConfig, SlicesCfg
from ..bias.mitigation import effective_sample_size, sample_weights
from ..bias.slices import build_slice_columns
from ..common.config import FeatureRegistry, FeatureSpec, MlConfig
from ..training.calibration import apply_platt, fit_platt, recent_months, sigmoid
from ..training.contracts import Candidate
from ..training.data import average_precision, to_model_frame
from ..training.ensemble import candidate_weights, combine
from ..training.members import LgbmMember, LrMember
from . import metrics, slices

VARIANTS = ("none", "reweight")


class MitigationVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: str
    pr_auc: float
    brier: float
    ece: float
    max_abs_oe: float | None  # over the gated, non-geographic slice values
    miscalibrated: int  # gated slice values whose O-E CI lies outside ±bias.oe_abs_max


class MitigationTradeoff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: str
    scheme: str
    weight_slice: str
    rows: int
    effective_sample_size: float
    variants: list[MitigationVariant]
    by_value: list[dict[str, Any]]  # slice, value, oe_none, oe_reweight (gated values)

    def variant(self, name: str) -> MitigationVariant:
        return next(v for v in self.variants if v.variant == name)


def eval_weights(
    frame: pl.DataFrame, bias_cfg: BiasConfig, registry: FeatureRegistry | None = None
) -> np.ndarray:
    """Weights for every row of an ML training frame, fit on that frame's own rows (leakage test 9)."""
    only = bias_cfg.model_copy(
        update={"slices": SlicesCfg(single=[bias_cfg.mitigation.weight_slice])}
    )
    with_slices = build_slice_columns(frame, only, registry)
    w = sample_weights(with_slices, bias_cfg, fit=pl.Series([True] * frame.height))
    return w["weight"].cast(pl.Float64).to_numpy()


def weighted_oof(
    frame: pl.DataFrame,
    features: Sequence[FeatureSpec],
    cfg: MlConfig,
    candidate: Candidate,
    weights: np.ndarray,
) -> np.ndarray:
    """The candidate's calibrated OOF probability with members fit on ``weights``: the folds of ``prepare``, each
    member fit on the other folds, the ensemble's log-odds mean, and Platt on the newest months (as ``train``)."""
    x = to_model_frame(frame, features)
    y = frame[f"y_{cfg.label}"].cast(pl.Float64).to_numpy()
    folds = frame["fold"].to_numpy()
    members = candidate_weights(candidate, cfg.ensemble.weights)
    oof: dict[str, np.ndarray] = {}
    for m in members:
        pred = np.full(len(y), np.nan)
        for k in np.unique(folds):
            held = folds == k
            xt, yt, wt = x[~held].reset_index(drop=True), y[~held], weights[~held]
            member: LgbmMember | LrMember = (
                LgbmMember.fit(xt, yt, features, cfg.members.lgbm, sample_weight=wt)
                if m == "lgbm"
                else LrMember.fit(xt, yt, features, cfg.members.lr, sample_weight=wt)
            )
            pred[held] = member.logit(x[held].reset_index(drop=True))
        oof[m] = pred
    raw = combine(oof, members)
    months = recent_months(frame["t"].to_list(), cfg.calibration.fit_recent_months)
    mask = frame["t"].is_in(months).to_numpy()
    return sigmoid(apply_platt(fit_platt(raw[mask], y[mask], months), raw))


def _variant(
    name: str, y: np.ndarray, p: np.ndarray, table: pl.DataFrame, cfg: MlConfig
) -> MitigationVariant:
    g = slices.gated(table, cfg)
    return MitigationVariant(
        variant=name,
        pr_auc=average_precision(y, p),
        brier=float(np.mean((p - y) ** 2)),
        ece=metrics.ece(y, p, cfg.evaluation.ece_bins),
        max_abs_oe=float(g["oe"].abs().max()) if g.height else None,  # type: ignore[arg-type]
        miscalibrated=slices.calibration_failures(table, cfg).height,
    )


def tradeoff(
    frame: pl.DataFrame,
    features: Sequence[FeatureSpec],
    cfg: MlConfig,
    candidate: Candidate,
    p_none: np.ndarray,
    *,
    bias_cfg: BiasConfig,
    registry: FeatureRegistry | None = None,
) -> MitigationTradeoff:
    """Unweighted (``p_none``, ``train``'s OOF) vs reweighted OOF predictions of ``candidate`` on the eval frame."""
    y = frame[f"y_{cfg.label}"].cast(pl.Float64).to_numpy()
    w = eval_weights(frame, bias_cfg, registry)
    p_rw = weighted_oof(frame, features, cfg, candidate, w)
    gate = slices.gated_slices(cfg)
    tables = {
        name: slices.slice_table(frame, y, p, slices=gate, bias_cfg=bias_cfg, registry=registry)
        for name, p in (("none", p_none), ("reweight", p_rw))
    }
    gated = {k: slices.gated(t, cfg).select("slice", "value", "oe") for k, t in tables.items()}
    by_value = (
        gated["none"]
        .rename({"oe": "oe_none"})
        .join(
            gated["reweight"].rename({"oe": "oe_reweight"}),
            on=["slice", "value"],
            how="full",
            coalesce=True,
        )
        .sort("slice", "value")
    )
    return MitigationTradeoff(
        candidate=candidate,
        scheme=bias_cfg.mitigation.weights,
        weight_slice=bias_cfg.mitigation.weight_slice,
        rows=frame.height,
        effective_sample_size=effective_sample_size(pl.Series(w)),
        variants=[
            _variant(k, y, p, tables[k], cfg) for k, p in (("none", p_none), ("reweight", p_rw))
        ],
        by_value=by_value.to_dicts(),
    )
