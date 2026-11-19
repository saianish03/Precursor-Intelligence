"""evaluation/promote.py: manual promotion and rollback (docs/ML_PIPELINE_SPEC.md §9.4, §17.1 manual-promote rules)."""

import shutil

import pytest
from ml_eval_fixtures import FakeRegistry

from precursorintelligence.evaluation.gates import GateReport, GateResult
from precursorintelligence.evaluation.packages import copy_packages, stable_prefix
from precursorintelligence.evaluation.promote import (
    PromoteUsageError,
    request_from_env,
    run_promote,
)
from precursorintelligence.training import MlInputError
from precursorintelligence.training.contracts import Champion, RunPaths
from precursorintelligence.training.ensemble import PACKAGE_META


def gates(**failed):
    """A gate report: G1-G7 pass unless named, e.g. ``gates(G2=True)``."""
    return GateReport(
        results=[
            GateResult(gate=g, status="fail" if failed.get(g) else "pass", reason="")
            for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
        ]
    ).model_dump_json()


def registry(run, cls=FakeRegistry):
    """Versions 1-3, each registered by a gate row of this run in ml_runs (its packages copied to the package
    store, as gate does). Version 3 failed G2 at its gate. The MLflow tags all say sanity_passed=true: promote
    must not trust them (§9.4)."""
    copy_packages(run.store(), run.packages(), run.cfg, run.run_id)
    reg = cls()
    gate_results = {1: gates(), 2: gates(), 3: gates(G2=True, G4=True)}
    for v in gate_results:
        reg.register(
            f"models:/v{v}",
            {"ml_run_id": run.run_id, "sanity_passed": "true", "recipe_hash": f"h{v}"},
        )
    run.meta().append(
        "ml_runs",
        [
            {
                "ml_run_id": run.run_id,
                "action": "hold",
                "candidate_version": v,
                "gate_results_json": results,
            }
            for v, results in gate_results.items()
        ],
    )
    return reg


def copied_refit_file(run):
    return f"{stable_prefix(run.cfg, run.run_id)}refit/{run.train.result.selected}/{PACKAGE_META}"


def promote(run, reg, version, reason="Checkpoint ML-A approved"):
    return run_promote(
        run.store(),
        run.meta(),
        packages=run.packages(),
        ml_run_id=run.run_id,
        cfg=run.cfg,
        registry=reg,
        version=version,
        reason=reason,
    )


def promote_rows(run):
    return [r for r in run.meta().read("ml_runs") if r["action"] in ("manual_promote", "rollback")]


def test_the_first_promotion_moves_champion_and_is_audited(trained):
    reg = registry(trained)
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
    reg = registry(trained)
    reg.set_alias("challenger", 2)
    promote(trained, reg, 2)
    assert reg.by_alias("challenger") is None and reg.by_alias("champion").version == 2


def test_an_earlier_version_is_a_rollback(trained):
    reg = registry(trained)
    reg.set_alias("champion", 2)
    out = promote(trained, reg, 1, reason="v2 scores looked wrong")
    assert (out.action, out.champion_before) == ("rollback", 2)
    assert reg.by_alias("champion").version == 1
    assert promote_rows(trained)[0]["action"] == "rollback"


def test_only_versions_that_passed_g1_and_g2_can_be_promoted(trained):
    reg = registry(trained)
    reg.set_alias("champion", 1)
    with pytest.raises(MlInputError, match="G1 and G2"):
        promote(trained, reg, 3)
    assert reg.by_alias("champion").version == 1 and not promote_rows(trained)


def test_the_g1_g2_check_reads_ml_runs_not_the_mlflow_tag(trained):
    # §9.4: MLflow has no login, so its sanity_passed tag can say anything; the gate row in ml_runs decides
    reg = registry(trained)
    assert reg.get(3).tags["sanity_passed"] == "true"
    with pytest.raises(MlInputError, match="failed G2"):
        promote(trained, reg, 3)
    assert reg.by_alias("champion") is None and not promote_rows(trained)


def test_a_version_no_gate_registered_is_refused(trained):
    reg = registry(trained)
    reg.register("models:/v4", {"ml_run_id": "elsewhere", "sanity_passed": "true"})
    with pytest.raises(MlInputError, match="no gate run that registered version 4"):
        promote(trained, reg, 4)
    assert reg.by_alias("champion") is None and not promote_rows(trained)


@pytest.mark.parametrize("how", ["altered", "deleted"])
def test_a_version_whose_package_fails_verification_is_refused(trained, how):
    # B4: a rollback to a version that cannot score must not move the alias
    reg = registry(trained)
    reg.set_alias("champion", 2)
    key = copied_refit_file(trained)
    if how == "altered":
        with trained.packages().open_read(key) as f:
            data = f.read()
        trained.packages().put_bytes(key, data + b" ", overwrite=True)
        match = "does not match the sha256"
    else:
        (trained.gold.root / "packages" / key).unlink()
        match = "is missing"
    with pytest.raises(MlInputError, match=match):
        promote(trained, reg, 1)
    assert reg.by_alias("champion").version == 2 and not promote_rows(trained)


def test_a_rollback_works_after_the_versions_run_folder_packages_are_deleted(trained):
    # §9.4 (#75 B17): promote verifies the copy in the package store, not the run folder
    reg = registry(trained)
    reg.set_alias("champion", 2)
    p = RunPaths.of(trained.cfg, trained.run_id)
    shutil.rmtree(trained.gold.root / f"{p.run_dir}models")
    (trained.gold.root / p.train_result).unlink()
    assert promote(trained, reg, 1).action == "rollback"
    assert reg.by_alias("champion").version == 1


def test_an_unknown_version_is_refused(trained):
    with pytest.raises(MlInputError, match="no version 9"):
        promote(trained, registry(trained), 9)


def test_promote_needs_the_registry(trained):
    with pytest.raises(PromoteUsageError, match="registry"):
        promote(trained, None, 1)


def test_a_retried_promote_writes_one_row(trained):
    reg = registry(trained)
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


class AliasFailsOnce(FakeRegistry):
    def __init__(self):
        super().__init__()
        self.armed = False

    def set_alias(self, alias, version):
        if self.armed:
            self.armed = False
            raise RuntimeError("MLflow went away")
        super().set_alias(alias, version)


def test_a_retried_rollback_still_records_a_rollback(trained):
    # the audit row is written before the alias moves, so a retry cannot read the moved alias as "before"
    reg = AliasFailsOnce()
    registry(trained, cls=lambda: reg)
    reg.set_alias("champion", 2)
    reg.armed = True
    with pytest.raises(RuntimeError):
        promote(trained, reg, 1)
    assert reg.by_alias("champion").version == 2  # not moved yet, and the row says what will happen
    out = promote(trained, reg, 1)
    assert (out.action, out.champion_before) == ("rollback", 2)
    (row,) = promote_rows(trained)
    assert (row["action"], row["champion_version_before"], row["champion_version_after"]) == (
        "rollback",
        2,
        1,
    )
    assert reg.by_alias("champion").version == 1
