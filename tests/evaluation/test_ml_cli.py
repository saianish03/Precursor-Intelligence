"""`ml evaluate`, `ml gate` and `ml promote` through the CLI (docs/ML_PIPELINE_SPEC.md §15.3, §15.5)."""

import json
import sys

import mlflow
import pytest
import yaml
from make_ml_gold import make_ml_gold
from ml_eval_fixtures import small_config

from precursorintelligence.harmonize.cli import main


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["precursor-pipeline", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


def last_json(capsys):
    out = capsys.readouterr().out
    return json.loads(out.strip().splitlines()[-1]), out


def common(gold, cfg_path, run_id):
    return ["--env-file", str(gold.env_file), "--run-id", run_id, "--ml-config", str(cfg_path)]


def write_config(tmp_path, cfg):
    path = tmp_path / "ml.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    return path


def test_evaluate_then_gate_without_tracking(trained, monkeypatch, capsys, tmp_path):
    args = common(trained.gold, write_config(tmp_path, trained.cfg), trained.run_id)
    assert run(monkeypatch, "ml", "evaluate", *args, "--no-shuffled-label") == 0
    summary, _ = last_json(capsys)
    assert summary["selected"] == trained.train.result.selected and summary["test_pr_auc"] > 0
    assert run(monkeypatch, "ml", "gate", *args) == 0  # a hold is exit 0
    summary, out = last_json(capsys)
    assert summary["action"] == "hold" and summary["version"] is None
    assert "G7" in out


def test_gate_before_evaluate_is_a_data_refusal(trained, monkeypatch, capsys, tmp_path):
    args = common(trained.gold, write_config(tmp_path, trained.cfg), trained.run_id)
    assert run(monkeypatch, "ml", "gate", *args) == 2
    assert "ml evaluate" in capsys.readouterr().err


def test_promote_without_a_version_is_a_usage_error(trained, monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("PI_ML_VERSION", raising=False)
    args = common(trained.gold, write_config(tmp_path, trained.cfg), trained.run_id)
    assert run(monkeypatch, "ml", "promote", *args) == 3
    assert "PI_ML_VERSION" in capsys.readouterr().err


def test_the_whole_chain_registers_and_promotes_in_mlflow(tmp_path, monkeypatch, capsys, tracking):
    """prepare -> train -> evaluate -> gate (first run: challenger) -> promote (Checkpoint ML-A), on a throwaway
    MLflow store."""
    gold = make_ml_gold(tmp_path / "store")
    cfg = small_config()
    cal = cfg.gates.calibration.model_copy(update={"ece_max": 0.5})  # tiny fixture models
    cfg = cfg.model_copy(update={"gates": cfg.gates.model_copy(update={"calibration": cal})})
    args = common(gold, write_config(tmp_path, cfg), "manual__chain")
    for step in ("prepare", "train", "evaluate", "gate"):
        assert run(monkeypatch, "ml", step, *args) == 0, capsys.readouterr()
    gate, _ = last_json(capsys)
    assert gate["passed"], gate
    assert (gate["action"], gate["version"], gate["alias"]) == ("hold", 1, "challenger")

    client = mlflow.tracking.MlflowClient()
    v = client.get_model_version(cfg.registry.model_name, "1")
    assert v.tags["ml_run_id"] == "manual__chain" and v.tags["eval_model_uri"]
    runs = client.search_runs(
        [mlflow.get_experiment_by_name(cfg.registry.experiment).experiment_id],
        "tags.ml_run_id = 'manual__chain'",
    )
    assert {r.data.tags["task"] for r in runs} == {"train", "evaluate", "gate"}
    evaluate_run = next(r for r in runs if r.data.tags["task"] == "evaluate")
    assert "test_pr_auc" in evaluate_run.data.metrics

    monkeypatch.setenv("PI_ML_VERSION", "1")
    monkeypatch.setenv("PI_ML_REASON", "Checkpoint ML-A: card reviewed")
    assert run(monkeypatch, "ml", "promote", *args) == 0
    promoted, _ = last_json(capsys)
    assert promoted == {
        "ml_run_id": "manual__chain",
        "action": "manual_promote",
        "champion_version": 1,
        "champion_version_before": None,
    }
    champion = client.get_model_version_by_alias(cfg.registry.model_name, "champion")
    assert str(champion.version) == "1" and champion.aliases == ["champion"]  # no longer challenger
