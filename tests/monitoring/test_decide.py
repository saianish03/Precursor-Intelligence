"""monitoring/decide (§15.2): the decision table row by row, then run_decide and `ml decide` on the synthetic gold."""

import json
import sys

import pytest
from make_ml_gold import ml_config

from precursorintelligence.common.config import load_feature_registry
from precursorintelligence.harmonize.cli import main
from precursorintelligence.monitoring.decide import decision_for, run_decide
from precursorintelligence.training import GoldPinError
from precursorintelligence.training.contracts import RunPaths

GOLD = {"v": "2026-10", "v_labels": "2026-10", "dataset_version": "dv1", "verified": True}
RECIPE, GOLD_SHA = "r1", "g1"


def gate_row(id_, decision="RETRAIN", **kw):
    return {
        "id": id_,
        "action": "hold",
        "decision": decision,
        "v_labels": "2026-10",
        "recipe_hash": RECIPE,
        "gold_config_sha": GOLD_SHA,
        **kw,
    }


def published_row(id_, version=1, v="2026-10", dataset_version="dv1"):
    return {
        "id": id_,
        "action": "score",
        "scores_published": True,
        "v": v,
        "dataset_version": dataset_version,
        "champion_version_after": version,
    }


def decide(rows=(), mode="auto", champion=1, last_scored=None, allow=True, **gold):
    args = {**GOLD, **gold}
    return decision_for(
        mode=mode,
        allow_unverified=allow,
        recipe_hash=RECIPE,
        gold_sha=GOLD_SHA,
        champion=champion,
        last_scored=last_scored,
        rows=list(rows),
        **args,
    )[0]


TRAINED = [gate_row(1)]
DONE = [gate_row(1), published_row(2)]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"mode": "promote", "rows": DONE}, "PROMOTE"),
        (
            {"rows": DONE, "last_scored": "2026-11"},
            "REFUSE",
        ),  # an older gold must not overwrite newer scores
        ({"rows": DONE, "verified": False, "allow": False}, "REFUSE"),
        ({"rows": DONE, "verified": False, "allow": True}, "SKIP"),
        ({"rows": DONE, "mode": "retune"}, "RETUNE"),
        (
            {"rows": [*DONE, {"id": 3, "action": "monitor", "retune_requested": True}]},
            "SKIP",
        ),  # a drift request asks a human: Phase 1 never re-tunes on its own
        ({"rows": DONE, "mode": "retrain"}, "RETRAIN"),
        ({"rows": []}, "RETRAIN"),  # first run
        ({"rows": [gate_row(1, recipe_hash="old")]}, "RETRAIN"),  # new code or config
        ({"rows": [gate_row(1, gold_config_sha="old")]}, "RETRAIN"),  # gold built differently
        ({"rows": [gate_row(1, v_labels="2026-09")]}, "RETRAIN"),  # labels matured a month
        ({"rows": TRAINED}, "SCORE"),  # a failed publish: score again
        ({"rows": [gate_row(1), published_row(2, version=2)]}, "SCORE"),  # champion changed
        (
            {"rows": [gate_row(1), published_row(2, dataset_version="dv0")]},
            "SCORE",
        ),  # stale refresh
        ({"rows": DONE, "mode": "score_only"}, "SCORE"),
        (
            {"rows": [gate_row(1, recipe_hash="old")], "mode": "score_only"},
            "SCORE",
        ),  # explicit mode wins
        (
            {"rows": TRAINED, "champion": None},
            "SKIP",
        ),  # a hold and no champion: awaiting Checkpoint ML-A
        ({"rows": DONE}, "SKIP"),
    ],
)
def test_the_decision_table(kwargs, expected):
    assert decide(**kwargs) == expected


def test_a_hold_then_a_no_op_gold_publish_skips_instead_of_retraining():
    assert decide(rows=[gate_row(1, action="hold")], champion=None) == "SKIP"


# --------------------------------------------------------------------------- on the synthetic gold


def run(store, meta, run_id="manual__d", **kw):
    return run_decide(
        store, meta, ml_run_id=run_id, cfg=ml_config(), registry=load_feature_registry(), **kw
    )


def test_a_first_run_retrains_and_records_the_gold_it_decided_on(gold, store, meta):
    d = run(store, meta)
    assert (d.decision, d.proceed, d.gold_run_id, d.v) == ("RETRAIN", True, gold.run_id, gold.v)
    paths = RunPaths.of(ml_config(), "manual__d")
    assert json.loads(store.open_read(paths.decision).read())["decision"] == "RETRAIN"
    assert meta.read("ml_runs") == []  # gate writes a retraining run's row


def test_score_writes_the_recorded_champion(store, meta):
    meta.append("ml_runs", [{"ml_run_id": "old", "action": "promote", "champion_version_after": 3}])
    d = run(store, meta, run_id="manual__s", mode="score_only")
    assert d.decision == "SCORE" and d.champion_version == 3
    champion = json.loads(store.open_read(RunPaths.of(ml_config(), "manual__s").champion).read())
    assert (champion["version"], champion["source"]) == (3, "decide")


def test_retune_proceeds_and_writes_its_row_once(store, meta):
    for _ in range(2):  # a retried decide
        d = run(store, meta, run_id="manual__t", mode="retune")
        assert (d.decision, d.proceed) == ("RETUNE", True)
    rows = meta.read("ml_runs", {"ml_run_id": "manual__t"})
    assert [(r["action"], r["mode"]) for r in rows] == [("retune", "retune")]


def test_refuse_writes_its_row_once(store, meta):
    meta.append(
        "ml_runs", [{"ml_run_id": "x", "action": "score", "scores_published": True, "v": "2099-01"}]
    )
    for _ in range(2):  # a retried decide
        assert run(store, meta, run_id="manual__r").decision == "REFUSE"
    rows = meta.read("ml_runs", {"ml_run_id": "manual__r"})
    assert [(r["action"], r["decision"]) for r in rows] == [("refuse", "REFUSE")]


def test_gold_not_current_is_a_retried_error(gold, store, meta):
    current = gold.root / "gold" / "_current.json"
    current.write_text(json.dumps({**json.loads(current.read_text()), "state": "building"}))
    with pytest.raises(GoldPinError, match="not 'current'"):
        run(store, meta)


def cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["precursor-pipeline", "ml", "decide", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


def test_ml_decide_prints_the_decision_and_exits_2_on_refuse(gold, meta, monkeypatch, capsys):
    args = ["--env-file", str(gold.env_file), "--run-id", "manual__c"]
    assert cli(monkeypatch, *args) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (out["decision"], out["proceed"]) == ("RETRAIN", True)

    meta.append(
        "ml_runs", [{"ml_run_id": "x", "action": "score", "scores_published": True, "v": "2099-01"}]
    )
    assert cli(monkeypatch, "--env-file", str(gold.env_file), "--run-id", "manual__c2") == 2
    assert "older than the last scored" in capsys.readouterr().err


def test_ml_decide_refuses_promote_without_a_version_and_reason(gold, monkeypatch, capsys):
    """Exit 3 before prepare builds every frame, not at promote; nothing is decided."""
    args = ["--env-file", str(gold.env_file), "--run-id", "manual__p", "--mode", "promote"]
    monkeypatch.delenv("PI_ML_VERSION", raising=False)
    monkeypatch.delenv("PI_ML_REASON", raising=False)
    assert cli(monkeypatch, *args) == 3
    assert "PI_ML_VERSION" in capsys.readouterr().err
    assert not (gold.root / RunPaths.of(ml_config(), "manual__p").decision).exists()

    monkeypatch.setenv("PI_ML_VERSION", "1")
    monkeypatch.setenv("PI_ML_REASON", "Checkpoint ML-A approved")
    assert cli(monkeypatch, *args) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["decision"] == "PROMOTE"
    assert (gold.root / RunPaths.of(ml_config(), "manual__p").decision).exists()
