"""training/select.py: the selection rule after the bias check (docs/ML_PIPELINE_SPEC.md §4.3), table-driven."""

import importlib
import sys
import types

import numpy as np
import pytest

from precursorintelligence.training.data import average_precision
from precursorintelligence.training.select import (
    BiasOutcome,
    bias_checked,
    default_bias_check,
    select_candidate,
    unavailable_bias_check,
)

# 20 positives, 80 negatives
Y = np.array([1] * 20 + [0] * 80)
_rng = np.random.default_rng(3)
SCORES = {
    "perfect": np.where(Y == 1, 0.8, 0.2),  # AP 1.0
    # one negative ranked above the (tied) positives: AP = 20 / 21
    "good": np.where(Y == 1, 0.8, 0.2) + np.r_[np.zeros(20), [0.7], np.zeros(79)],
    "poor": _rng.random(100) * 0.9,  # roughly the base rate
}

PASS, FAIL = BiasOutcome(True), BiasOutcome(False, {"failing_slices": ["x"]})


def test_fixture_orders_as_intended():
    ap = {k: average_precision(Y.astype(float), v) for k, v in SCORES.items()}
    assert ap["perfect"] == pytest.approx(1.0)
    assert ap["good"] == pytest.approx(20 / 21)
    assert ap["poor"] < 0.5


CASES = [
    # (id, {candidate: score name}, failing candidates, tie_margin, expected, reason fragment)
    (
        "highest_wins",
        {"ensemble": "perfect", "lgbm": "good", "lr": "poor"},
        set(),
        0.0,
        "ensemble",
        "highest OOF PR-AUC",
    ),
    (
        "highest_wins_simple",
        {"ensemble": "good", "lgbm": "poor", "lr": "perfect"},
        set(),
        0.0,
        "lr",
        "highest OOF PR-AUC",
    ),
    (
        "simpler_within_margin",
        {"ensemble": "perfect", "lgbm": "good", "lr": "poor"},
        set(),
        0.2,
        "lgbm",
        "simpler candidate wins",
    ),
    (
        "simplest_within_wide_margin",
        {"ensemble": "perfect", "lgbm": "good", "lr": "poor"},
        set(),
        1.0,
        "lr",
        "simpler candidate wins",
    ),
    (
        "exact_tie_goes_to_simplest",
        {"ensemble": "perfect", "lgbm": "perfect", "lr": "perfect"},
        set(),
        0.0,
        "lr",
        "simpler candidate wins",
    ),
    (
        "best_fails_bias_is_dropped",
        {"ensemble": "perfect", "lgbm": "good", "lr": "poor"},
        {"ensemble"},
        0.0,
        "lgbm",
        "dropped by the bias check: ['ensemble']",
    ),
    (
        "dropped_even_within_margin",
        {"ensemble": "perfect", "lgbm": "good", "lr": "perfect"},
        {"lr"},
        0.2,
        "lgbm",
        "dropped by the bias check: ['lr']",
    ),
    (
        "all_fail_best_selected",
        {"ensemble": "perfect", "lgbm": "good", "lr": "poor"},
        {"ensemble", "lgbm", "lr"},
        0.0,
        "ensemble",
        "every candidate failed the OOF bias check",
    ),
    (
        "all_fail_tie_rule_still_applies",
        {"ensemble": "perfect", "lgbm": "good", "lr": "poor"},
        {"ensemble", "lgbm", "lr"},
        0.2,
        "lgbm",
        "is the simplest within 0.2 of the best OOF PR-AUC (ensemble",  # lgbm is not the best: say so
    ),
]


@pytest.mark.parametrize(
    "scores,failing,margin,expected,fragment", [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_selection_rule(scores, failing, margin, expected, fragment):
    oof = {c: SCORES[s] for c, s in scores.items()}
    bias = {c: FAIL if c in failing else PASS for c in scores}
    sel = select_candidate(Y, oof, bias, margin)
    assert sel.selected == expected
    assert fragment in sel.reason
    by = {r.candidate: r for r in sel.candidates}
    assert set(by) == set(scores)
    for c, r in by.items():
        assert r.bias_check_passed is (c not in failing)
        assert r.oof_pr_auc == pytest.approx(average_precision(Y.astype(float), oof[c]), abs=1e-6)
        assert r.oof_brier == pytest.approx(float(np.mean((oof[c] - Y) ** 2)), abs=1e-6)
    if failing == set(scores):
        assert not any(r.bias_check_passed for r in sel.candidates)
    for c in failing:
        assert "failing_slices" in by[c].note


def test_candidates_restrict_the_pool():
    oof = {c: SCORES["perfect"] for c in ("ensemble", "lgbm", "lr")}
    bias = dict.fromkeys(oof, PASS)
    sel = select_candidate(Y, oof, bias, 0.0, candidates=["ensemble", "lgbm"])
    assert sel.selected == "lgbm"
    assert [r.candidate for r in sel.candidates] == ["ensemble", "lgbm"]


# ------------------------------------------------------------------ default_bias_check


SLICES = "precursorintelligence.evaluation.slices"


def _hide_slices(monkeypatch, module):
    import precursorintelligence.evaluation as evaluation

    monkeypatch.delattr(evaluation, "slices", raising=False)
    monkeypatch.setitem(sys.modules, SLICES, module)


def _import_raises(monkeypatch, exc):
    real = importlib.import_module

    def fake(name, *args, **kwargs):
        if name == SLICES:
            raise exc
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake)


def test_default_bias_check_without_w2_is_a_pass_through(monkeypatch):
    _hide_slices(monkeypatch, None)  # import fails with ModuleNotFoundError(name=SLICES)
    fn = default_bias_check()
    assert fn is unavailable_bias_check
    out = fn(None, np.zeros(3), None)
    assert out.passed is True
    assert "not available" in out.detail["note"]


@pytest.mark.parametrize("missing", [SLICES, "precursorintelligence.evaluation"])
def test_default_bias_check_falls_back_only_for_the_w2_modules(monkeypatch, missing):
    _import_raises(monkeypatch, ModuleNotFoundError(f"No module named {missing!r}", name=missing))
    assert default_bias_check() is unavailable_bias_check


def test_default_bias_check_module_without_the_function_raises(monkeypatch):
    _hide_slices(monkeypatch, types.ModuleType(SLICES))
    with pytest.raises(AttributeError, match="bias_check_oof"):
        default_bias_check()


def test_default_bias_check_propagates_a_missing_dependency_of_slices(monkeypatch):
    # slices.py exists but its own import fails: a bug in W2's module, not "W2 not merged"
    _import_raises(
        monkeypatch, ModuleNotFoundError("No module named 'fairlearn'", name="fairlearn")
    )
    with pytest.raises(ModuleNotFoundError, match="fairlearn"):
        default_bias_check()


def test_default_bias_check_propagates_other_import_errors(monkeypatch):
    _import_raises(monkeypatch, ImportError("cannot import name 'x' from 'polars'", name="polars"))
    with pytest.raises(ImportError, match="cannot import name"):
        default_bias_check()


def test_default_bias_check_uses_w2_when_present(monkeypatch):
    mod = types.ModuleType(SLICES)

    def bias_check_oof(rows, p, cfg):
        return BiasOutcome(False)

    mod.bias_check_oof = bias_check_oof
    _hide_slices(monkeypatch, mod)
    assert default_bias_check() is bias_check_oof


def test_reason_says_when_the_bias_check_did_not_run():
    unchecked = unavailable_bias_check(None, np.zeros(3), None)
    sel = select_candidate(
        Y,
        {"lr": SCORES["poor"], "lgbm": SCORES["perfect"]},
        {"lr": unchecked, "lgbm": unchecked},
        tie_margin=0.002,
    )
    assert sel.selected == "lgbm"
    assert "bias check not run" in sel.reason
    assert not any(bias_checked(r) for r in sel.candidates)


def test_reason_after_a_real_bias_check():
    sel = select_candidate(
        Y, {"lr": SCORES["poor"], "lgbm": SCORES["perfect"]}, {"lr": PASS, "lgbm": PASS}, 0.002
    )
    assert sel.reason.endswith("after the bias check")
    assert all(bias_checked(r) for r in sel.candidates)
