"""#73 B16: in prod an ML task without MLflow tracking refuses (exit 3) instead of running and registering nothing."""

import sys

import pytest
import yaml

from precursorintelligence.evidence import build as evidence_build
from precursorintelligence.harmonize.cli import main
from precursorintelligence.ingestion.pipeline import EXIT_CONFIG, EXIT_VALIDATION
from precursorintelligence.training import MlInputError
from precursorintelligence.training import tune as tune_mod


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["precursor-pipeline", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


def _env_file(gold, tmp_path, env):
    data = yaml.safe_load(gold.env_file.read_text())
    data["env"] = env
    path = tmp_path / f"{env}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


@pytest.fixture
def body_ran(monkeypatch):
    """The task bodies refuse with exit 2, so exit 2 means the guard let the task start."""

    def refuse(*args, **kwargs):
        raise MlInputError("the body ran")

    monkeypatch.setattr(tune_mod, "run_tune", refuse)
    monkeypatch.setattr(evidence_build, "run_build_evidence", refuse)


@pytest.mark.parametrize(
    ("env", "uri", "expected"),
    [
        ("prod", None, EXIT_CONFIG),
        ("prod", "http://mlflow:5000", EXIT_VALIDATION),
        ("dev", None, EXIT_VALIDATION),
    ],
)
def test_an_ml_task_in_prod_needs_tracking(
    gold, tmp_path, monkeypatch, capsys, body_ran, env, uri, expected
):
    if uri:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    else:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    code = _run(
        monkeypatch,
        "ml",
        "tune",
        "--env-file",
        str(_env_file(gold, tmp_path, env)),
        "--run-id",
        "x",
    )
    assert code == expected
    if expected == EXIT_CONFIG:
        assert "MLFLOW_TRACKING_URI" in capsys.readouterr().err


def test_build_evidence_in_prod_runs_without_tracking(gold, tmp_path, monkeypatch, body_ran):
    # it reads ml_runs and the run folders, never MLflow
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    env_file = _env_file(gold, tmp_path, "prod")
    assert (
        _run(monkeypatch, "agents", "build-evidence", "--v", "2026-10", "--env-file", str(env_file))
        == EXIT_VALIDATION
    )
