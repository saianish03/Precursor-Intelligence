"""monitoring/status.run_monitor (§12): signals, status/latest.json, the monitor row and the re-tune request."""

import json

import pytest

from precursorintelligence.common.config import load_feature_registry
from precursorintelligence.evaluation.evaluate import run_evaluate
from precursorintelligence.monitoring.decide import decision_for
from precursorintelligence.monitoring.status import run_monitor
from precursorintelligence.serving.score import score_report_dir
from precursorintelligence.training.contracts import RunPaths


def monitor(trained, cfg=None):
    return run_monitor(
        trained.store(),
        trained.meta(),
        ml_run_id=trained.run_id,
        cfg=cfg or trained.cfg,
        registry=load_feature_registry(),
    )


def status_file(trained):
    cfg = trained.cfg
    return json.loads(
        trained.store().open_read(f"{cfg.outputs.root_prefix}{cfg.outputs.status_key}").read()
    )


def with_key_features(trained, names):
    key = f"{score_report_dir(RunPaths.of(trained.cfg, trained.run_id))}key_features.json"
    trained.store().put_bytes(key, json.dumps(names).encode(), overwrite=True)


def test_without_a_champion_the_status_says_awaiting_checkpoint_ml_a(trained):
    out = monitor(trained)
    status = status_file(trained)
    assert status == out.status
    assert status["model_version"] is None and status["n_facilities"] == 0
    assert status["note"] == "no champion yet: awaiting Checkpoint ML-A"
    assert status["candidate"] == trained.train.result.selected
    assert "drift" not in out.psi  # no scores this run, so no key features
    # no scores, so no key features: skew covers the enforcement and regime-sensitive features (§12.1)
    enforcement = {f.name for f in load_feature_registry().family("enforcement")}
    assert set(out.psi["skew"]) <= enforcement | set(trained.cfg.features.regime_sensitive)
    assert enforcement & set(out.psi["skew"])


def test_a_drifting_key_feature_asks_a_human_to_retune_and_auto_never_does(trained):
    with_key_features(trained, ["sev_points_12m", "hprd_total"])
    strict = trained.cfg.model_copy(
        update={"monitor": trained.cfg.monitor.model_copy(update={"feature_psi_max": -1.0})}
    )
    out = monitor(trained, strict)
    assert set(out.psi["drift"]) == {"sev_points_12m", "hprd_total"}
    assert {"drift:sev_points_12m", "drift:hprd_total"} <= set(out.status["flags"])
    assert out.status["retune_requested"] is True

    rows = trained.meta().read("ml_runs")
    [row] = [r for r in rows if r["action"] == "monitor"]
    assert row["retune_requested"] is True
    decision, _ = decision_for(
        mode="auto",
        v="2026-10",
        v_labels="2026-10",
        dataset_version="dv",
        verified=True,
        allow_unverified=True,
        recipe_hash="r",
        gold_sha="g",
        champion=None,
        last_scored=None,
        rows=rows,
    )
    assert decision != "RETUNE"  # Phase 1: only mode=retune re-tunes, so a request can't loop


def test_skew_alone_alerts_without_a_retune(trained):
    strict = trained.cfg.model_copy(
        update={"monitor": trained.cfg.monitor.model_copy(update={"feature_psi_max": -1.0})}
    )
    out = monitor(trained, strict)
    assert any(f.startswith("skew:") for f in out.status["flags"])
    assert out.status["retune_requested"] is False


def test_a_retried_monitor_writes_its_row_once(trained):
    monitor(trained)
    monitor(trained)
    rows = trained.meta().read("ml_runs", {"ml_run_id": trained.run_id, "action": ["monitor"]})
    assert len(rows) == 1


def test_a_champion_trained_long_ago_is_flagged_stale(trained):
    trained.meta().append(
        "ml_runs",
        [
            {
                "ml_run_id": "old",
                "action": "promote",
                "candidate_version": 4,
                "v_labels": "2025-01",
                "champion_version_after": 4,
            },
        ],
    )
    out = monitor(trained)
    assert out.status["model_version"] == 4
    assert any(f.startswith("stale_champion:") for f in out.status["flags"])


# --------------------------------------------------------------------------- score drift (§12.1)


def with_score_drift(trained, **drift):
    report = {"prev_month": "2026-09", "n_prev": 100, "n": 100, "mean_prev": 0.25, "mean": 0.25}
    report.update({"psi": 0.0, "mean_shift": 0.0, **drift})
    key = f"{score_report_dir(RunPaths.of(trained.cfg, trained.run_id))}score_drift.json"
    trained.store().put_bytes(key, json.dumps(report).encode(), overwrite=True)
    return report


@pytest.mark.parametrize(
    ("drift", "flags"),
    [
        ({"psi": 0.1, "mean_shift": 0.03}, set()),  # at the limits: not over them
        ({"psi": 0.1001}, {"score_drift:psi"}),
        ({"mean_shift": 0.0301}, {"score_drift:mean"}),
        ({"mean_shift": -0.0301}, {"score_drift:mean"}),
        ({"psi": 0.5, "mean_shift": -0.05}, {"score_drift:psi", "score_drift:mean"}),
    ],
)
def test_score_drift_flags_over_its_limits_and_asks_for_a_retune(trained, drift, flags):
    report = with_score_drift(trained, **drift)
    out = monitor(trained)
    assert {f for f in out.status["flags"] if f.startswith("score_drift:")} == flags
    assert out.status["score_drift"] == report
    assert out.status["retune_requested"] is bool(flags)
    assert status_file(trained)["score_drift"] == report


def test_without_scores_this_run_there_is_no_score_drift(trained):
    out = monitor(trained)
    assert out.status["score_drift"] is None
    assert not any(f.startswith("score_drift:") for f in out.status["flags"])


def test_no_previous_month_is_reported_without_a_flag(trained):
    with_score_drift(trained, prev_month=None, n_prev=0, psi=None, mean_prev=None, mean_shift=None)
    out = monitor(trained)
    assert out.status["score_drift"]["prev_month"] is None
    assert not any(f.startswith("score_drift:") for f in out.status["flags"])


# --------------------------------------------------------------------------- performance trend (§12.1)


def _evaluate(trained, development_months):
    cfg = trained.cfg.model_copy(
        update={
            "evaluation": trained.cfg.evaluation.model_copy(
                update={"development_months": development_months}
            )
        }
    )
    run_evaluate(trained.store(), ml_run_id=trained.run_id, cfg=cfg, shuffled_label=False)
    return trained


@pytest.fixture
def evaluated(trained):
    """The trained run after ``evaluate``, its test month an honest one (not a development month)."""
    return _evaluate(trained, [])


@pytest.fixture
def evaluated_on_development(trained):
    """The same, with the test month (2025-02) a development month, as configs/ml.yaml has it (§3.2)."""
    return _evaluate(trained, trained.cfg.evaluation.development_months)


def this_run_months(trained):
    report = json.loads(
        trained.store().open_read(RunPaths.of(trained.cfg, trained.run_id).evaluation).read()
    )
    return [m for m in report["per_month"] if m["pr_auc_lift"] is not None]


def earlier_run(trained, run_id, lifts, development=()):
    """An earlier evaluated run: a gate row in ml_runs and an evaluation.json with these months' lifts."""
    meta = trained.meta()
    meta.append("ml_runs", [{"ml_run_id": run_id, "action": "hold", "candidate_version": 9}])
    per_month = [
        {"month": m, "pr_auc_lift": lift, "development": m in development}
        for m, lift in lifts.items()
    ]
    trained.store().put_bytes(
        RunPaths.of(trained.cfg, run_id).evaluation,
        json.dumps({"per_month": per_month}).encode(),
        overwrite=True,
    )


def months_before(month, n):
    y, m = int(month[:4]), int(month[5:7])
    out = []
    for _ in range(n):
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
        out.append(f"{y:04d}-{m:02d}")
    return out


@pytest.mark.parametrize(("ratio", "flagged"), [(0.84, True), (0.86, False)])
def test_a_lift_more_than_15_percent_under_the_earlier_median_is_flagged(evaluated, ratio, flagged):
    [this] = this_run_months(evaluated)
    median = this["pr_auc_lift"] / ratio
    a, b, c = months_before(this["month"], 3)
    earlier_run(evaluated, "old-1", {a: median - 0.2, b: median})
    earlier_run(evaluated, "old-2", {c: median + 0.2})
    out = monitor(evaluated)
    trend = out.status["performance_trend"][this["month"]]
    assert trend["n_earlier"] == 3 and trend["median_earlier"] == pytest.approx(median)
    assert trend["flagged"] is flagged
    assert (f"performance:lift_drop:{this['month']}" in out.status["flags"]) is flagged
    assert out.status["retune_requested"] is flagged


def test_the_latest_run_counts_for_a_month_and_later_months_never(evaluated):
    [this] = this_run_months(evaluated)
    [a] = months_before(this["month"], 1)
    earlier_run(evaluated, "old-1", {a: 100.0})
    earlier_run(evaluated, "old-2", {a: this["pr_auc_lift"], this["month"]: 100.0})
    out = monitor(evaluated)
    trend = out.status["performance_trend"][this["month"]]
    assert trend["n_earlier"] == 1 and trend["median_earlier"] == pytest.approx(this["pr_auc_lift"])
    assert trend["flagged"] is False


def test_no_earlier_months_reports_the_lift_without_a_flag(evaluated):
    [this] = this_run_months(evaluated)
    out = monitor(evaluated)
    assert out.status["performance_trend"] == {
        this["month"]: {
            "lift": pytest.approx(this["pr_auc_lift"]),
            "n_earlier": 0,
            "median_earlier": None,
            "development": False,
            "flagged": False,
        }
    }
    assert not any(f.startswith("performance:") for f in out.status["flags"])


def test_a_development_month_never_enters_the_baseline(evaluated):
    # the recipe was chosen on a development month (§3.2), so its lift is optimistic and would flag honest months
    [this] = this_run_months(evaluated)
    a, b = months_before(this["month"], 2)
    earlier_run(evaluated, "old-1", {a: this["pr_auc_lift"], b: 100.0}, development={b})
    trend = monitor(evaluated).status["performance_trend"][this["month"]]
    assert trend["n_earlier"] == 1 and trend["median_earlier"] == pytest.approx(this["pr_auc_lift"])
    assert trend["flagged"] is False


def test_a_development_month_of_this_run_is_reported_never_flagged(evaluated_on_development):
    [this] = this_run_months(evaluated_on_development)
    [a] = months_before(this["month"], 1)
    earlier_run(evaluated_on_development, "old-1", {a: 100.0})
    out = monitor(evaluated_on_development)
    trend = out.status["performance_trend"][this["month"]]
    assert trend["development"] is True and trend["n_earlier"] == 1
    assert trend["flagged"] is False
    assert not any(f.startswith("performance:") for f in out.status["flags"])
    assert out.status["retune_requested"] is False


def test_a_run_that_did_not_evaluate_has_no_performance_trend(trained):
    out = monitor(trained)
    assert out.status["performance_trend"] is None


# --------------------------------------------------------------------------- skew scope (#73 B20)


def test_skew_without_key_features_says_it_is_partial(trained):
    assert (
        monitor(trained).status["skew_scope"] == "partial: no key features (this run didn't score)"
    )


def test_skew_with_key_features_is_full(trained):
    with_key_features(trained, ["sev_points_12m"])
    assert monitor(trained).status["skew_scope"] == "full"
