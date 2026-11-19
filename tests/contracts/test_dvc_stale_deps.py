"""scripts/dvc_stale_deps.py: CI fails a PR whose toy stages' inputs changed without a dvc.lock refresh (#73 B14)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dvc_stale_deps.py"


def load_script():
    spec = importlib.util.spec_from_file_location("dvc_stale_deps", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_changed_dependency_or_command_is_stale():
    status = {
        "toy_ml": [{"changed deps": {"src/precursorintelligence/validation": "modified"}}],
        "toy_gold": [{"changed command": True}],
    }
    assert load_script().stale_stages(status) == ["toy_gold", "toy_ml"]


def test_missing_outputs_alone_are_not_stale():
    # a fresh checkout has no data/: CI rebuilds the outputs, and metrics drift stays a warning. A stage that reads
    # another's output sees it as a deleted dependency (toy_summary on toy_gold's gold, as on CI's first run)
    status = {
        "toy_gold": [{"changed outs": {"data/toy/gold": "not in cache"}}],
        "toy_summary": [{"changed deps": {"data/toy/gold/spine": "deleted"}}],
    }
    assert load_script().stale_stages(status) == []


def test_a_modified_dependency_next_to_a_missing_one_is_stale():
    status = {"toy_ml": [{"changed deps": {"data/toy/gold": "deleted", "uv.lock": "modified"}}]}
    assert load_script().stale_stages(status) == ["toy_ml"]


def test_it_exits_1_and_names_the_stale_stages():
    run = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=json.dumps({"toy_ml": [{"changed deps": {"uv.lock": "modified"}}]}),
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1 and run.stdout.split() == ["toy_ml"]
    up_to_date = subprocess.run(
        [sys.executable, str(SCRIPT)], input="{}", capture_output=True, text=True
    )
    assert up_to_date.returncode == 0 and up_to_date.stdout == ""
