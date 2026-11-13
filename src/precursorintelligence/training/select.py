"""Candidate selection after the bias check (docs/ML_PIPELINE_SPEC.md §4.3, ML-28).

On the eval window's out-of-fold predictions only (never the test frame):
1. drop every candidate that fails the OOF bias check G7(b) (W2's ``bias_check_oof``);
   until W2's ``evaluation/slices.py`` exists the check is a pass-through that says so (``default_bias_check``);
2. keep the highest OOF PR-AUC; a simpler candidate wins when it is within ``selection.tie_margin`` of the best
   (complexity: lr < lgbm < ensemble).

If every candidate fails the bias check, the best one by the same rule is still selected, marked as failing, so the
run reaches ``gate``, where G7 holds it (a hold, not a crash).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ..common.config import MlConfig
from .calibration import brier
from .contracts import Candidate, CandidateResult, Selection
from .data import average_precision

log = logging.getLogger(__name__)

COMPLEXITY: dict[Candidate, int] = {"lr": 0, "lgbm": 1, "ensemble": 2}


@dataclass(frozen=True)
class BiasOutcome:
    """W2's answer for one candidate: ``passed`` and anything worth recording (the failing slices)."""

    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)


# (OOF rows with their gold columns and ``fold``, calibrated OOF probability, the ML config) -> outcome.
BiasCheck = Callable[[pl.DataFrame, np.ndarray, MlConfig], BiasOutcome]


BIAS_NOT_CHECKED = "bias_check_oof not available (W2 not merged): not checked"


def unavailable_bias_check(rows: pl.DataFrame, p: np.ndarray, cfg: MlConfig) -> BiasOutcome:
    return BiasOutcome(passed=True, detail={"note": BIAS_NOT_CHECKED})


def bias_checked(result: CandidateResult) -> bool:
    """False when the pass-through ran instead of W2's check (``bias_check_passed`` then means nothing)."""
    return BIAS_NOT_CHECKED not in result.note


SLICES_MODULE = "precursorintelligence.evaluation.slices"
# The only modules whose absence means "W2 not merged" (the package itself, or its slices module).
_W2_MISSING = frozenset({SLICES_MODULE, SLICES_MODULE.rpartition(".")[0]})


def default_bias_check() -> BiasCheck:
    """W2's ``evaluation.slices.bias_check_oof``; a pass-through that says so only while that module does not exist.

    The fallback is for W2 not merged, nothing else: a ``ModuleNotFoundError`` for any other module (one that
    ``evaluation/slices.py`` itself imports) propagates, and a ``slices`` module without ``bias_check_oof`` raises
    ``AttributeError``. Both are bugs, and skipping G7(b) silently would hide them."""
    try:
        module = importlib.import_module(SLICES_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name not in _W2_MISSING:
            raise
        log.warning(
            "%s does not exist (W2 not merged): selection skips the bias check", SLICES_MODULE
        )
        return unavailable_bias_check
    if not hasattr(module, "bias_check_oof"):
        raise AttributeError(f"{SLICES_MODULE} exists but has no bias_check_oof")
    fn: BiasCheck = module.bias_check_oof
    return fn


def select_candidate(
    y: np.ndarray,
    oof_p: Mapping[Candidate, np.ndarray],
    bias: Mapping[Candidate, BiasOutcome],
    tie_margin: float,
    candidates: Sequence[Candidate] | None = None,
) -> Selection:
    """Apply the rule to OOF probabilities (calibrated; PR-AUC is the same on raw log-odds)."""
    names: list[Candidate] = list(candidates or oof_p.keys())
    results = []
    for c in names:
        note = "; ".join(f"{k}: {v}" for k, v in bias[c].detail.items())
        results.append(
            CandidateResult(
                candidate=c,
                oof_pr_auc=round(average_precision(np.asarray(y, float), oof_p[c]), 6),
                oof_brier=round(brier(y, oof_p[c]), 6),
                bias_check_passed=bias[c].passed,
                note=note,
            )
        )
    passing = [r for r in results if r.bias_check_passed]
    pool = passing or results
    best = max(r.oof_pr_auc for r in pool)
    within = [r for r in pool if r.oof_pr_auc >= best - tie_margin]
    chosen = min(within, key=lambda r: COMPLEXITY[r.candidate])
    top = max(pool, key=lambda r: r.oof_pr_auc)
    if not passing:
        reason = (
            f"every candidate failed the OOF bias check; {chosen.candidate} (OOF PR-AUC {chosen.oof_pr_auc}) is "
            f"the simplest within {tie_margin} of the best OOF PR-AUC ({top.candidate}, {top.oof_pr_auc}) and is "
            "selected so that gate G7 holds it"
        )
    elif chosen.candidate == top.candidate:
        after = (
            "after the bias check"
            if any(bias_checked(r) for r in results)
            else "(bias check not run: W2's bias_check_oof not merged)"
        )
        reason = f"{chosen.candidate}: highest OOF PR-AUC ({chosen.oof_pr_auc}) {after}"
    else:
        reason = (
            f"{chosen.candidate}: OOF PR-AUC {chosen.oof_pr_auc} within {tie_margin} of {top.candidate} "
            f"({top.oof_pr_auc}); the simpler candidate wins the tie"
        )
    dropped = [r.candidate for r in results if not r.bias_check_passed]
    if dropped and passing:
        reason += f"; dropped by the bias check: {dropped}"
    return Selection(selected=chosen.candidate, candidates=results, reason=reason)
