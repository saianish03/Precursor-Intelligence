"""The promotion gates G1-G7 (docs/ML_PIPELINE_SPEC.md §9.1): deterministic pass/fail rules on numbers the
evaluation already computed. Promotion is this code, not judgment (§0.4).

Every gate returns ``pass``, ``fail`` or ``skip`` (it has nothing to judge, e.g. G5 with no champion), a reason
and the values it used. A version is promoted only if no gate fails; any failure is a hold (§9.1).

"Worse" than the champion means beyond the tolerance **and** the paired 95 % interval of the difference
excludes zero, so a retrain is never held on noise (ML-27). Differences are always new minus reference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..common.config import MlConfig

Status = Literal["pass", "fail", "skip"]
Value = float | int | str | bool | None


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Interval(_Model):
    """A point estimate with its paired-bootstrap interval (``BootstrapResult.difference``)."""

    estimate: float
    ci_low: float
    ci_high: float


class GateResult(_Model):
    gate: str
    status: Status
    reason: str
    values: dict[str, Value] = {}


def _result(gate: str, failures: Sequence[str], ok: str, values: Mapping[str, Value]) -> GateResult:
    if failures:
        return GateResult(gate=gate, status="fail", reason="; ".join(failures), values=dict(values))
    return GateResult(gate=gate, status="pass", reason=ok, values=dict(values))


def g1_data(*, gold_verified: bool, allow_unverified: bool) -> GateResult:
    """Input checks passed (``prepare`` refuses otherwise, so a run that got here passed them) and gold is
    verified, or unverified gold is allowed (ML-9)."""
    fails = [] if gold_verified or allow_unverified else ["gold is unverified and not allowed"]
    return _result(
        "G1",
        fails,
        "input checks passed; gold verified"
        if gold_verified
        else "input checks passed; unverified gold allowed",
        {"gold_verified": gold_verified, "allow_unverified": allow_unverified},
    )


def g2_sanity(
    *,
    uncleared_leakage_flags: Sequence[str],
    shap_additive: bool | None,
    shuffled_label_roc: float | None,
    roc_range: tuple[float, float],
) -> GateResult:
    """No uncleared single-feature leakage flag; SHAP contributions add up (when checked); the shuffled-label
    model's ROC-AUC is near 0.5 (when it ran)."""
    fails = []
    if uncleared_leakage_flags:
        fails.append(f"uncleared leakage flags: {sorted(uncleared_leakage_flags)}")
    if shap_additive is False:
        fails.append("SHAP contributions do not add up to the model's log-odds")
    lo, hi = roc_range
    if shuffled_label_roc is not None and not lo <= shuffled_label_roc <= hi:
        fails.append(f"shuffled-label ROC-AUC {shuffled_label_roc:.4f} outside [{lo}, {hi}]")
    return _result(
        "G2",
        fails,
        "no leakage flag; sanity checks in range",
        {
            "uncleared_leakage_flags": len(uncleared_leakage_flags),
            "shap_additive": shap_additive,
            "shuffled_label_roc": shuffled_label_roc,
        },
    )


def g3_beats_rule(diff: Interval) -> GateResult:
    """The selected candidate's PR-AUC beats the prior-G+ rule's: the paired interval of the difference is
    above zero."""
    values = {"pr_auc_diff": diff.estimate, "ci_low": diff.ci_low, "ci_high": diff.ci_high}
    fails = (
        []
        if diff.ci_low > 0
        else [
            f"PR-AUC vs the rule {diff.estimate:+.4f}, interval [{diff.ci_low:+.4f}, {diff.ci_high:+.4f}] not above 0"
        ]
    )
    return _result("G3", fails, f"beats the prior-G+ rule by {diff.estimate:+.4f} PR-AUC", values)


def g4_calibration(*, ece: float, slope: float, cfg: MlConfig) -> GateResult:
    """Test ECE at most ``gates.calibration.ece_max`` and the calibration slope inside ``slope_range``."""
    c = cfg.gates.calibration
    lo, hi = c.slope_range
    fails = []
    if not ece <= c.ece_max:
        fails.append(f"ECE {ece:.4f} > {c.ece_max}")
    if not lo <= slope <= hi:
        fails.append(f"calibration slope {slope:.3f} outside [{lo}, {hi}]")
    return _result("G4", fails, f"ECE {ece:.4f}, slope {slope:.3f}", {"ece": ece, "slope": slope})


def g5_vs_champion(
    *, pr_auc_diff: Interval | None, brier_diff: Interval | None, cfg: MlConfig
) -> GateResult:
    """Not worse than the champion's eval fit on the same test rows: PR-AUC not lower by more than
    ``pr_auc_worse_max``, Brier not higher by more than ``brier_worse_max``, each significant. Skipped with no
    champion."""
    if pr_auc_diff is None or brier_diff is None:
        return GateResult(gate="G5", status="skip", reason="no champion to compare with")
    v = cfg.gates.vs_champion
    fails = []
    if pr_auc_diff.estimate < -v.pr_auc_worse_max and pr_auc_diff.ci_high < 0:
        fails.append(
            f"PR-AUC {pr_auc_diff.estimate:+.4f} vs the champion (interval [{pr_auc_diff.ci_low:+.4f}, {pr_auc_diff.ci_high:+.4f}])"
        )
    if brier_diff.estimate > v.brier_worse_max and brier_diff.ci_low > 0:
        fails.append(
            f"Brier {brier_diff.estimate:+.4f} vs the champion (interval [{brier_diff.ci_low:+.4f}, {brier_diff.ci_high:+.4f}])"
        )
    return _result(
        "G5",
        fails,
        "not worse than the champion",
        {"pr_auc_diff": pr_auc_diff.estimate, "brier_diff": brier_diff.estimate},
    )


def g6_refit(
    *,
    refit_oof_lift: float,
    eval_oof_lift: float,
    score_mean_p: float,
    recent_observed_rate: float,
    spearman: float,
    cfg: MlConfig,
) -> GateResult:
    """The model that ships looks like the one evaluated: its OOF lift is at least ``oof_lift_ratio_min`` of the
    eval fit's; its mean score this month is within ``mean_p_band`` of the refit window's newest observed rate;
    and it ranks this month's homes like the eval fit (Spearman at least ``spearman_min``)."""
    r = cfg.gates.refit
    ratio = refit_oof_lift / eval_oof_lift if eval_oof_lift else float("nan")
    fails = []
    if not ratio >= r.oof_lift_ratio_min:
        fails.append(f"refit OOF lift is {ratio:.3f} of the eval fit's (< {r.oof_lift_ratio_min})")
    if not abs(score_mean_p - recent_observed_rate) <= r.mean_p_band:
        fails.append(
            f"mean score {score_mean_p:.3f} vs recent rate {recent_observed_rate:.3f} (band ±{r.mean_p_band})"
        )
    if not spearman >= r.spearman_min:
        fails.append(f"Spearman of refit and eval-fit scores {spearman:.3f} < {r.spearman_min}")
    return _result(
        "G6",
        fails,
        f"lift ratio {ratio:.3f}, mean score {score_mean_p:.3f}, Spearman {spearman:.3f}",
        {
            "oof_lift_ratio": ratio,
            "score_mean_p": score_mean_p,
            "recent_observed_rate": recent_observed_rate,
            "spearman": spearman,
        },
    )


class SliceComparison(_Model):
    """One gated slice value, new model minus the champion's eval fit on the same test rows."""

    slice: str
    value: str
    abs_oe_diff: Interval  # |O-E| new - |O-E| champion: positive = calibration got worse
    pr_auc_diff: Interval  # PR-AUC new - champion: negative = ranking got worse


def g7_bias(
    *,
    calibration_failures: Sequence[str],
    vs_champion: Sequence[SliceComparison] | None,
    cfg: MlConfig,
) -> GateResult:
    """(b) no gated slice value is clearly miscalibrated (``evaluation.slices.calibration_failures`` on the
    test frame); (a) against the champion, no gated slice value's |O-E| grows by more than ``oe_worse_max`` nor
    its PR-AUC drops by more than ``pr_auc_worse_max``, each significant. (a) is skipped with no champion."""
    b = cfg.bias
    fails = [f"miscalibrated: {f}" for f in calibration_failures]
    for s in vs_champion or []:
        name = f"{s.slice}={s.value}"
        if s.abs_oe_diff.estimate > b.oe_worse_max and s.abs_oe_diff.ci_low > 0:
            fails.append(f"{name}: |O-E| up {s.abs_oe_diff.estimate:+.3f} vs the champion")
        if s.pr_auc_diff.estimate < -b.pr_auc_worse_max and s.pr_auc_diff.ci_high < 0:
            fails.append(f"{name}: PR-AUC {s.pr_auc_diff.estimate:+.3f} vs the champion")
    part_a = (
        "skipped (no champion)"
        if vs_champion is None
        else f"{len(vs_champion)} slice values compared"
    )
    return _result(
        "G7",
        fails,
        f"no slice clearly miscalibrated; vs champion: {part_a}",
        {"calibration_failures": len(calibration_failures), "vs_champion": part_a},
    )


class GateReport(_Model):
    """Every gate's result; ``passed`` when none failed (skips don't block)."""

    schema_version: Literal[1] = 1
    results: list[GateResult]

    @property
    def passed(self) -> bool:
        return all(r.status != "fail" for r in self.results)

    @property
    def failed(self) -> list[str]:
        return [r.gate for r in self.results if r.status == "fail"]
