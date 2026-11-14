"""evaluation/promote.py: manual promotion and rollback (docs/ML_PIPELINE_SPEC.md §9.4, §17.1 manual-promote rules)."""

import pytest
from ml_eval_fixtures import FakeRegistry

from precursorintelligence.evaluation.promote import (
    PromoteUsageError,
    request_from_env,
    run_promote,
)
from precursorintelligence.training import MlInputError
from precursorintelligence.training.contracts import Champion, RunPaths


def registry():
    reg = FakeRegistry()
    reg.register("models:/a", {"ml_run_id": "r1", "sanity_passed": "true", "recipe_hash": "h1"})
    reg.register("models:/b", {"ml_run_id": "r2", "sanity_passed": "true", "recipe_hash": "h2"})
    reg.register(
        "models:/c", {"ml_run_id": "r3", "sanity_passed": "false", "failed_gates": "G2,G4"}
    )
    return reg


def promote(run, reg, version, reason="Checkpoint ML-A approved"):
    return run_promote(
        run.store(),
        run.meta(),
        ml_run_id=run.run_id,
        cfg=run.cfg,
        registry=reg,
        version=version,
        reason=reason,
    )


def promote_rows(run):
    return [r for r in run.meta().read("ml_runs") if r["action"] in ("manual_promote", "rollback")]


def test_the_first_promotion_moves_champion_and_is_audited(trained):
    reg = registry()
    out = promote(trained, reg, 2)
    assert (out.action, out.champion_before, out.version) == ("manual_promote", None, 2)
    assert reg.by_alias("champion").version == 2
    (row,) = promote_rows(trained)
    assert row["champion_version_after"] == 2 and row["reason"] == "Checkpoint ML-A approved"
    assert row["recipe_hash"] == "h2" and row["gold_run_id"]  # the PROMOTE run's prepared gold
    p = RunPaths.of(trained.cfg, trained.run_id)
    with trained.store().open_read(p.champion) as f:
        assert Champion.model_validate_json(f.read()).model_dump() == {
            "schema_version": 1,
            "model_name": trained.cfg.registry.model_name,
            "version": 2,
            "source": "promote",
        }


def test_promoting_the_challenger_clears_that_alias(trained):
    reg = registry()
    reg.set_alias("challenger", 2)
    promote(trained, reg, 2)
    assert reg.by_alias("challenger") is None and reg.by_alias("champion").version == 2


def test_an_earlier_version_is_a_rollback(trained):
    reg = registry()
    reg.set_alias("champion", 2)
    out = promote(trained, reg, 1, reason="v2 scores looked wrong")
    assert (out.action, out.champion_before) == ("rollback", 2)
    assert reg.by_alias("champion").version == 1
    assert promote_rows(trained)[0]["action"] == "rollback"


def test_only_versions_that_passed_g1_and_g2_can_be_promoted(trained):
    reg = registry()
    reg.set_alias("champion", 1)
    with pytest.raises(MlInputError, match="G1 and G2"):
        promote(trained, reg, 3)
    assert reg.by_alias("champion").version == 1 and not promote_rows(trained)


def test_an_unknown_version_is_refused(trained):
    with pytest.raises(MlInputError, match="no version 9"):
        promote(trained, registry(), 9)


def test_promote_needs_the_registry(trained):
    with pytest.raises(PromoteUsageError, match="registry"):
        promote(trained, None, 1)


def test_a_retried_promote_writes_one_row(trained):
    reg = registry()
    promote(trained, reg, 2)
    promote(trained, reg, 2)
    assert len(promote_rows(trained)) == 1


def test_the_request_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("PI_ML_VERSION", "4")
    monkeypatch.setenv("PI_ML_REASON", "rollback after a bad release")
    assert request_from_env() == (4, "rollback after a bad release")


@pytest.mark.parametrize(
    "version, reason, message",
    [
        ("", "x", "version number"),
        ("abc", "x", "version number"),
        ("0", "x", ">= 1"),
        ("2", " ", "required"),
    ],
)
def test_a_bad_request_is_a_usage_error(monkeypatch, version, reason, message):
    monkeypatch.setenv("PI_ML_VERSION", version)
    monkeypatch.setenv("PI_ML_REASON", reason)
    with pytest.raises(PromoteUsageError, match=message):
        request_from_env()
