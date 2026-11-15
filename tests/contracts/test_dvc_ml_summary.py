"""scripts/dvc_ml_summary.py: the toy ML smoke run's metrics file, and its refusal to pass a chain that skipped."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dvc_ml_summary.py"


def load_script():
    spec = importlib.util.spec_from_file_location("dvc_ml_summary", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_keeps_the_last_json_line_per_step_without_run_ids_or_timings(tmp_path):
    out = tmp_path / "train.out"
    out.write_text(
        'run toy: trained\n{"selected": "lr", "mlflow_run_id": "abc", "seconds_total": 3.2}\nbye\n'
    )
    assert load_script().summarize([out]) == {"train": {"selected": "lr"}}


def test_a_skipped_step_fails_the_smoke_run(tmp_path):
    out = tmp_path / "score.out"
    out.write_text('{"skipped": true, "reason": "no champion yet"}\n')
    with pytest.raises(SystemExit, match="score skipped"):
        load_script().summarize([out])
