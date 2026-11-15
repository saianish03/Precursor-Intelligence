"""scripts/dvc_toy_ml.py: the toy ML smoke run's metrics, and its refusal to pass a chain that held or skipped."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dvc_toy_ml.py"


def load_script():
    spec = importlib.util.spec_from_file_location("dvc_toy_ml", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_summary_is_the_last_json_line():
    text = 'run toy: trained\n{"selected": "lr"}\na log line after it\n'
    assert load_script().last_json_object(text) == {"selected": "lr"}


def test_the_metrics_keep_only_facts_that_are_the_same_on_every_machine():
    summary = {
        "selected": "lr",
        "rows": {"score": 2000},
        "passed": True,
        "test_pr_auc": 0.41,
        "mlflow_run_id": "abc",
        "seconds_total": 3.2,
        "dataset_version": "038dd3030e1d9f34",
        "recipe_hash": "4032ed",
    }
    assert load_script().stable(summary) == {
        "selected": "lr",
        "rows": {"score": 2000},
        "passed": True,
    }


def test_a_skipped_step_fails_the_smoke_run():
    with pytest.raises(SystemExit, match="score skipped"):
        load_script().check("score", {"skipped": True, "reason": "no champion yet"})


@pytest.mark.parametrize(
    "gate",
    [
        {"passed": False, "action": "hold", "failed_gates": ["G7"]},
        {"passed": True, "action": "hold", "failed_gates": []},  # e.g. a champion already exists
    ],
)
def test_a_gate_that_does_not_promote_fails_the_smoke_run(gate):
    with pytest.raises(SystemExit, match="did not promote"):
        load_script().check("gate", gate)


def test_a_promoting_gate_passes():
    load_script().check("gate", {"passed": True, "action": "promote", "failed_gates": []})
