"""``evaluation.json``, the evaluation's one result file (docs/ML_PIPELINE_SPEC.md §6.5), written by ``evaluate``
and read by ``gate``. It holds every number the gates judge, so ``gate`` recomputes nothing on the test frame.

Intervals are 95 % facility-bootstrap percentiles (§6.3); a comparison is ``new - reference`` on the same
resampled facilities. NaN (a metric that is undefined, e.g. PR-AUC of a slice with too few positives) is written
as null.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_serializer

from .gates import Interval, SliceComparison

Number = float | None


def _clean(value: Any) -> Any:
    """NaN and infinity -> None, recursively (JSON has no NaN)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScoreReport(_Report):
    """One score on the test frame: a candidate's eval fit, a baseline, or the champion's eval fit."""

    ranking: dict[str, Any]
    probability: dict[str, Any] | None = (
        None  # probabilities only (not the baselines' ranking scores)
    )
    classification: dict[str, Any] | None = None  # at the OOF max-F1 threshold
    within_state: dict[str, Any] | None = None
    intervals: dict[str, Interval] = {}  # headline metric -> estimate and CI

    @field_serializer("ranking", "probability", "classification", "within_state")
    def _nan_to_null(self, v: dict[str, Any] | None) -> Any:
        return _clean(v)


class ChampionRef(_Report):
    version: int
    eval_model_uri: str
    candidate: str | None = None


class Sanity(_Report):
    """G2's inputs (§6.4)."""

    uncleared_leakage_flags: list[str]
    shap_additive: bool | None
    shap_max_abs_error: Number
    shap_rows: int
    shuffled_label_roc: Number  # None when it did not run (an already-evaluated recipe_hash)


class EvaluationReport(_Report):
    schema_version: Literal[1] = 1
    ml_run_id: str
    gold_run_id: str
    dataset_version: str
    recipe_hash: str
    label: str
    selected: str
    gold_verified: bool
    allow_unverified: bool
    test_months: list[str]
    development_months: list[str]  # test months chosen on by earlier work (§3.2): reported, marked
    rows: int
    facilities: int
    positives: int
    threshold: float  # the selected candidate's OOF max-F1 cut-off (never chosen on test)
    candidates: dict[str, ScoreReport]
    baselines: dict[str, ScoreReport]
    champion: ChampionRef | None
    champion_eval: ScoreReport | None
    oof_pr_auc: dict[str, float]
    oof_test_gap: dict[str, Number]  # OOF PR-AUC - test PR-AUC per candidate
    per_month: list[dict[str, Any]]  # the selected candidate per test month
    reliability: dict[str, list[dict[str, Any]]]  # "raw" (before Platt) and "calibrated", selected
    comparisons: dict[str, Interval]  # "<metric>:<new>-<reference>"
    slices: list[dict[str, Any]]  # evaluation.slices.slice_table of the selected candidate
    calibration_failures: list[str]  # G7(b)
    slices_vs_champion: list[SliceComparison] | None  # G7(a); None without a champion
    top_features: list[dict[str, Any]]  # mean |contribution| on the test frame, top 20
    sanity: Sanity
    bootstrap_replicates: int
    flags: list[str]

    @field_serializer("oof_test_gap", "per_month", "reliability", "slices", "top_features")
    def _nan_to_null(self, v: Any) -> Any:
        return _clean(v)

    def comparison(self, metric: str, new: str, reference: str) -> Interval | None:
        return self.comparisons.get(comparison_key(metric, new, reference))


def comparison_key(metric: str, new: str, reference: str) -> str:
    return f"{metric}:{new}-{reference}"
