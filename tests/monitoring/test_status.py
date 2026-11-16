"""monitoring/status.run_monitor (§12): signals, status/latest.json, the monitor row and the re-tune loop."""

import json

from precursorintelligence.monitoring.decide import decision_for
from precursorintelligence.monitoring.status import run_monitor
from precursorintelligence.serving.score import score_report_dir
from precursorintelligence.training.contracts import RunPaths
from precursorintelligence.training.data import read_input_manifest


def monitor(trained, cfg=None):
    return run_monitor(
        trained.store(), trained.meta(), ml_run_id=trained.run_id, cfg=cfg or trained.cfg
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
    assert set(out.psi["skew"]) <= set(trained.cfg.features.regime_sensitive)


def test_a_drifting_key_feature_requests_a_retune_that_the_next_decide_runs(trained):
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
    assert decision == "RETUNE"


def test_a_retune_run_s_own_monitor_does_not_ask_again(trained):
    """The RETUNE run's monitor sees the same two months and the same PSI: it still flags the drift, but asking
    again would re-tune on every gold event until a new month arrives."""
    v = read_input_manifest(trained.store(), RunPaths.of(trained.cfg, trained.run_id)).v
    trained.meta().append(
        "ml_runs",
        [
            {"ml_run_id": "drifted", "action": "monitor", "v": v, "retune_requested": True},
            {"ml_run_id": "retune", "action": "hold", "decision": "RETUNE", "v": v},
        ],
    )
    with_key_features(trained, ["sev_points_12m", "hprd_total"])
    strict = trained.cfg.model_copy(
        update={"monitor": trained.cfg.monitor.model_copy(update={"feature_psi_max": -1.0})}
    )
    out = monitor(trained, strict)
    assert {"drift:sev_points_12m", "drift:hprd_total"} <= set(out.status["flags"])
    assert out.status["retune_requested"] is False
    decision, _ = decision_for(
        mode="auto",
        v=v,
        v_labels=v,
        dataset_version="dv",
        verified=True,
        allow_unverified=True,
        recipe_hash="r",
        gold_sha="g",
        champion=None,
        last_scored=None,
        rows=trained.meta().read("ml_runs"),
    )
    assert decision != "RETUNE"


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
