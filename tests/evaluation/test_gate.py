"""evaluation/gate.py on an evaluated synthetic run (docs/ML_PIPELINE_SPEC.md §9, §17.1 registration idempotency)."""

import json

import pytest
from conftest import FakeRegistry

from precursorintelligence.evaluation.evaluate import run_evaluate
from precursorintelligence.evaluation.gate import refit_check, run_gate
from precursorintelligence.evaluation.report import EvaluationReport
from precursorintelligence.training import MlInputError
from precursorintelligence.training.contracts import Champion, RunPaths, TrainResult
from precursorintelligence.training.data import gold_config_sha, read_input_manifest, read_windows

REFIT_URI = "models:/refit-fixture"
EVAL_URI = "models:/eval-fixture"


def loose(cfg, **gates):
    """The fixture's tiny models are not calibrated to production's ECE limit: loosen G4 for promotion tests."""
    cal = cfg.gates.calibration.model_copy(update={"ece_max": 0.5})
    return cfg.model_copy(
        update={"gates": cfg.gates.model_copy(update={"calibration": cal, **gates})}
    )


def paths(run):
    return RunPaths.of(run.cfg, run.run_id)


def with_logged_models(run):
    """train.json as a run with MLflow tracking would have written it (model URIs to register)."""
    store, p = run.store(), paths(run)
    with store.open_read(p.train_result) as f:
        train = TrainResult.model_validate_json(f.read())
    sel = train.selected
    train = train.model_copy(
        update={
            "refit_model_uri": REFIT_URI,
            "eval_model_uris": {**train.eval_model_uris, sel: EVAL_URI},
        }
    )
    store.put_bytes(p.train_result, train.model_dump_json().encode(), overwrite=True)
    return train


def evaluate(run, cfg=None, registry=None):
    run_evaluate(
        run.store(),
        ml_run_id=run.run_id,
        cfg=cfg or run.cfg,
        registry=registry,
        shuffled_label=False,
    )


def gate(run, cfg=None, registry=None, meta=None):
    return run_gate(
        run.store(), meta or run.meta(), ml_run_id=run.run_id, cfg=cfg or run.cfg, registry=registry
    )


def gate_rows(run):
    return [r for r in run.meta().read("ml_runs") if r["action"] in ("promote", "hold")]


def test_gate_without_tracking_records_the_outcome_and_registers_nothing(trained):
    evaluate(trained)
    out = gate(trained)
    assert [r.gate for r in out.gates.results] == ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    assert {r.gate: r.status for r in out.gates.results}["G5"] == "skip"  # no champion
    assert out.version is None and out.alias is None and out.action == "hold"

    store, p = trained.store(), paths(trained)
    with store.open_read(p.champion) as f:
        champ = Champion.model_validate_json(f.read())
    assert champ.version is None and champ.source == "gate"
    assert store.exists(f"{p.run_dir}report/model_card.md")

    (row,) = gate_rows(trained)
    assert row["ml_run_id"] == trained.run_id and row["action"] == "hold"
    assert row["gold_config_sha"] == gold_config_sha(read_input_manifest(store, p))
    assert row["recipe_hash"] == "test" and row["champion_version_after"] is None
    assert [g["gate"] for g in json.loads(row["gate_results_json"])["results"]][0] == "G1"


def test_the_first_run_is_a_challenger_even_when_every_gate_passes(trained):
    cfg = loose(trained.cfg)
    with_logged_models(trained)
    evaluate(trained, cfg)
    reg = FakeRegistry()
    out = gate(trained, cfg, reg)
    assert out.gates.passed, out.gates.failed
    assert (out.version, out.alias, out.action) == (1, "challenger", "hold")
    assert reg.by_alias("champion") is None
    tags = reg.get(1).tags
    assert tags["ml_run_id"] == trained.run_id and tags["eval_model_uri"] == EVAL_URI
    assert tags["gate_outcome"] == "pass" and tags["sanity_passed"] == "true"
    assert reg.sources[1] == REFIT_URI
    assert "first run" in gate_rows(trained)[0]["reason"]


def test_first_run_promote_is_for_the_toy_smoke_run(trained):
    cfg = loose(trained.cfg, first_run="promote")
    with_logged_models(trained)
    evaluate(trained, cfg)
    reg = FakeRegistry()
    out = gate(trained, cfg, reg)
    assert (out.action, out.alias, out.champion_after) == ("promote", "champion", 1)


def test_a_retried_gate_reuses_its_version_and_row(trained):
    cfg = loose(trained.cfg)
    with_logged_models(trained)
    evaluate(trained, cfg)
    reg = FakeRegistry()
    first, again = gate(trained, cfg, reg), gate(trained, cfg, reg)
    assert first.version == again.version == 1 and list(reg.versions) == [1]
    assert len(gate_rows(trained)) == 1


def champion_registry(run, model_uri=EVAL_URI):
    """A registry whose champion (version 1) is this run's own selected eval fit."""
    chosen = run.train.result.selected
    reg = FakeRegistry({model_uri: run.train.eval_models[chosen]})
    reg.register("models:/old-refit", {"ml_run_id": "older", "eval_model_uri": model_uri})
    reg.set_alias("champion", 1)
    return reg


def test_passing_every_gate_against_a_champion_promotes(trained):
    cfg = loose(trained.cfg)
    with_logged_models(trained)
    reg = champion_registry(trained)
    evaluate(trained, cfg, reg)
    out = gate(trained, cfg, reg)
    status = {r.gate: r.status for r in out.gates.results}
    assert status["G5"] == "pass" and "compared" in out.gates.results[-1].reason
    assert (out.action, out.version, out.alias) == ("promote", 2, "champion")
    assert (out.champion_before, out.champion_after) == (1, 2)
    row = gate_rows(trained)[0]
    assert (row["champion_version_before"], row["champion_version_after"]) == (1, 2)


def test_a_failed_gate_holds_and_the_champion_stays(trained):
    with_logged_models(trained)
    reg = champion_registry(trained)
    strict = trained.cfg.model_copy(
        update={
            "gates": trained.cfg.gates.model_copy(
                update={
                    "calibration": trained.cfg.gates.calibration.model_copy(update={"ece_max": 0.0})
                }
            )
        }
    )
    evaluate(trained, strict, reg)
    out = gate(trained, strict, reg)
    assert "G4" in out.gates.failed
    assert (out.action, out.alias, out.champion_after) == ("hold", "challenger", 1)
    assert reg.by_alias("champion").version == 1
    assert reg.get(2).tags["failed_gates"] == "G4"
    assert gate_rows(trained)[0]["reason"].startswith("held: failed G4")


def test_registering_needs_the_refit_logged_to_mlflow(trained):
    evaluate(trained)
    with pytest.raises(MlInputError, match="nothing to register"):
        gate(trained, registry=FakeRegistry())


def test_gate_refuses_an_evaluation_of_another_recipe(trained):
    evaluate(trained)
    store, p = trained.store(), paths(trained)
    with store.open_read(p.evaluation) as f:
        report = EvaluationReport.model_validate_json(f.read())
    other = report.model_copy(update={"recipe_hash": "other"})
    store.put_bytes(p.evaluation, other.model_dump_json().encode(), overwrite=True)
    with pytest.raises(MlInputError, match="another run or recipe"):
        gate(trained)


def test_gate_needs_an_evaluated_run(trained):
    with pytest.raises(MlInputError, match="ml evaluate"):
        gate(trained)


def test_the_refit_check_compares_the_shipped_model_with_the_evaluated_one(trained):
    store, p = trained.store(), paths(trained)
    with store.open_read(p.train_result) as f:
        train = TrainResult.model_validate_json(f.read())
    rc = refit_check(store, p, train, read_windows(store, p), trained.cfg)
    assert rc.refit_oof_lift > 1 and rc.eval_oof_lift > 1  # the fixture plants a signal
    assert 0 < rc.score_mean_p < 1 and 0 < rc.recent_observed_rate < 1
    assert 0.5 < rc.spearman <= 1
