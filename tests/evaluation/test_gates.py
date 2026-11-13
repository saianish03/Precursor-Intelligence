"""evaluation/gates.py: each gate passes and fails at its boundary (docs/ML_PIPELINE_SPEC.md §9.1, §17.1)."""

import pytest

from precursorintelligence.common.config import load_ml_config
from precursorintelligence.evaluation.gates import (
    GateReport,
    GateResult,
    Interval,
    SliceComparison,
    g1_data,
    g2_sanity,
    g3_beats_rule,
    g4_calibration,
    g5_vs_champion,
    g6_refit,
    g7_bias,
)

CFG = (
    load_ml_config()
)  # ece_max 0.05, slope [0.7, 1.3], PR-AUC -0.01, Brier +0.005, refit 0.85 / 0.05 / 0.9


def iv(estimate: float, low: float, high: float) -> Interval:
    return Interval(estimate=estimate, ci_low=low, ci_high=high)


@pytest.mark.parametrize(
    "verified, allowed, status",
    [(True, False, "pass"), (False, True, "pass"), (False, False, "fail")],
)
def test_g1_needs_verified_or_allowed_gold(verified, allowed, status):
    assert g1_data(gold_verified=verified, allow_unverified=allowed).status == status


@pytest.mark.parametrize(
    "flags, shap, roc, status",
    [
        ([], True, 0.50, "pass"),
        ([], None, None, "pass"),  # checks that did not run don't fail the gate
        ([], True, 0.48, "pass"),
        ([], True, 0.52, "pass"),
        (["n_cit_12m"], True, 0.50, "fail"),
        ([], False, 0.50, "fail"),
        ([], True, 0.53, "fail"),
        ([], True, 0.47, "fail"),
    ],
)
def test_g2_sanity(flags, shap, roc, status):
    r = g2_sanity(
        uncleared_leakage_flags=flags,
        shap_additive=shap,
        shuffled_label_roc=roc,
        roc_range=(0.48, 0.52),
    )
    assert r.status == status


@pytest.mark.parametrize(
    "diff, status",
    [
        (iv(0.16, 0.146, 0.173), "pass"),
        (iv(0.01, 0.0001, 0.02), "pass"),
        (iv(0.01, 0.0, 0.02), "fail"),  # the interval must be strictly above zero
        (iv(0.01, -0.001, 0.02), "fail"),
        (iv(-0.05, -0.07, -0.03), "fail"),
    ],
)
def test_g3_needs_a_significant_win_over_the_rule(diff, status):
    assert g3_beats_rule(diff).status == status


@pytest.mark.parametrize(
    "ece, slope, status",
    [
        (0.028, 0.835, "pass"),
        (0.05, 0.7, "pass"),  # both limits are inclusive
        (0.05, 1.3, "pass"),
        (0.0501, 1.0, "fail"),
        (0.02, 0.699, "fail"),
        (0.02, 1.301, "fail"),
    ],
)
def test_g4_calibration_limits(ece, slope, status):
    assert g4_calibration(ece=ece, slope=slope, cfg=CFG).status == status


def test_g4_names_every_failure():
    r = g4_calibration(ece=0.08, slope=0.5, cfg=CFG)
    assert "ECE" in r.reason and "slope" in r.reason


def test_g5_is_skipped_without_a_champion():
    assert g5_vs_champion(pr_auc_diff=None, brier_diff=None, cfg=CFG).status == "skip"


@pytest.mark.parametrize(
    "pr_auc, brier, status",
    [
        (iv(0.002, -0.003, 0.007), iv(0.0, -0.001, 0.001), "pass"),
        (
            iv(-0.02, -0.03, 0.001),
            iv(0.0, -0.001, 0.001),
            "pass",
        ),  # beyond tolerance, but not significant
        (
            iv(-0.008, -0.012, -0.004),
            iv(0.0, -0.001, 0.001),
            "pass",
        ),  # significant, but within tolerance
        (
            iv(-0.02, -0.03, -0.01),
            iv(0.0, -0.001, 0.001),
            "fail",
        ),  # beyond tolerance and significant
        (iv(0.0, -0.005, 0.005), iv(0.006, 0.001, 0.011), "fail"),
        (iv(0.0, -0.005, 0.005), iv(0.006, -0.001, 0.013), "pass"),
        (
            iv(0.0, -0.005, 0.005),
            iv(0.005, 0.001, 0.009),
            "pass",
        ),  # exactly the tolerance is allowed
    ],
)
def test_g5_worse_needs_both_tolerance_and_significance(pr_auc, brier, status):
    assert g5_vs_champion(pr_auc_diff=pr_auc, brier_diff=brier, cfg=CFG).status == status


def g6(**kw):
    args = dict(
        refit_oof_lift=1.9,
        eval_oof_lift=2.0,
        score_mean_p=0.25,
        recent_observed_rate=0.26,
        spearman=0.95,
    )
    return g6_refit(**(args | kw), cfg=CFG)


@pytest.mark.parametrize(
    "kw, status",
    [
        ({}, "pass"),
        ({"refit_oof_lift": 1.7}, "pass"),  # 0.85 of 2.0
        ({"refit_oof_lift": 1.69}, "fail"),
        ({"score_mean_p": 0.31}, "pass"),  # 0.05 above 0.26
        ({"score_mean_p": 0.3101}, "fail"),
        ({"score_mean_p": 0.2099}, "fail"),
        ({"spearman": 0.9}, "pass"),
        ({"spearman": 0.894}, "fail"),
        ({"eval_oof_lift": 0.0}, "fail"),  # a degenerate eval fit can't vouch for the refit
    ],
)
def test_g6_refit_sanity(kw, status):
    assert g6(**kw).status == status


def comparison(oe: Interval, pr: Interval) -> SliceComparison:
    return SliceComparison(
        slice="ownership_group", value="for_profit", abs_oe_diff=oe, pr_auc_diff=pr
    )


@pytest.mark.parametrize(
    "failures, vs, status",
    [
        ([], None, "pass"),
        (["bed_band=200+: O-E +0.080 [+0.060, +0.100]"], None, "fail"),
        ([], [comparison(iv(0.01, -0.01, 0.03), iv(-0.01, -0.03, 0.01))], "pass"),
        ([], [comparison(iv(0.04, 0.01, 0.07), iv(0.0, -0.02, 0.02))], "fail"),
        ([], [comparison(iv(0.04, -0.01, 0.09), iv(0.0, -0.02, 0.02))], "pass"),  # not significant
        ([], [comparison(iv(0.0, -0.01, 0.01), iv(-0.06, -0.09, -0.03))], "fail"),
        ([], [comparison(iv(0.0, -0.01, 0.01), iv(-0.06, -0.09, 0.001))], "pass"),
    ],
)
def test_g7_bias(failures, vs, status):
    assert g7_bias(calibration_failures=failures, vs_champion=vs, cfg=CFG).status == status


def test_g7_says_when_the_champion_comparison_is_skipped():
    assert "skipped" in g7_bias(calibration_failures=[], vs_champion=None, cfg=CFG).reason


def test_the_report_passes_unless_a_gate_fails():
    ok = GateResult(gate="G1", status="pass", reason="")
    skip = GateResult(gate="G5", status="skip", reason="")
    bad = GateResult(gate="G4", status="fail", reason="")
    assert GateReport(results=[ok, skip]).passed
    report = GateReport(results=[ok, bad, skip])
    assert not report.passed and report.failed == ["G4"]


def test_the_report_round_trips_through_json():
    report = GateReport(results=[g4_calibration(ece=0.03, slope=1.0, cfg=CFG)])
    assert GateReport.model_validate_json(report.model_dump_json()) == report
