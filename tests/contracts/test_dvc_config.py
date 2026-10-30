"""The DVC setup is consistent: dvc.yaml is well-formed, its inputs exist, its outputs stay under data/, the
remote is the dedicated DVC bucket, and Git can see the files DVC needs committed. Needs no dvc install."""

import configparser
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
REMOTE_URL = "gs://precursor-intel-mlops-dvc"


@pytest.fixture(scope="module")
def stages():
    return yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]


def _paths(entries):
    """Paths of deps/outs/metrics; an entry is a string or a one-key mapping {path: options}."""
    return [e if isinstance(e, str) else next(iter(e)) for e in entries or []]


def test_every_stage_has_a_command_and_declares_what_it_produces(stages):
    assert stages
    for name, stage in stages.items():
        assert stage.get("cmd"), name
        assert _paths(stage.get("outs")) or _paths(stage.get("metrics")), name


def test_dependencies_that_are_not_stage_outputs_exist_in_the_repo(stages):
    produced = {
        p for s in stages.values() for p in _paths(s.get("outs")) + _paths(s.get("metrics"))
    }
    for name, stage in stages.items():
        for dep in _paths(stage.get("deps")):
            if dep not in produced:
                assert (ROOT / dep).exists(), f"{name}: dep {dep} is missing"


def test_the_locked_library_versions_are_a_declared_input(stages):
    # Parquet files record the writer version, so a polars/pyarrow bump changes every output hash.
    assert "uv.lock" in _paths(stages["toy_gold"]["deps"])


def test_data_outputs_stay_under_data_and_metrics_are_committed_to_git(stages):
    for name, stage in stages.items():
        for out in _paths(stage.get("outs")):
            assert out.startswith("data/"), f"{name}: {out} must live under data/ (git-ignored)"
        for entry in stage.get("metrics") or []:
            assert isinstance(entry, dict) and entry[_paths([entry])[0]]["cache"] is False, (
                f"{name}: metrics must be cache: false so Git tracks them"
            )


def test_the_default_remote_is_the_dedicated_dvc_bucket():
    config = configparser.ConfigParser()
    config.read(ROOT / ".dvc" / "config", encoding="utf-8")
    assert config["core"]["remote"] == "gcs"
    assert config["'remote \"gcs\"'"]["url"] == REMOTE_URL


def test_dvc_lock_covers_every_stage_so_the_data_versions_are_committed(stages):
    lock = yaml.safe_load((ROOT / "dvc.lock").read_text(encoding="utf-8"))["stages"]
    assert set(lock) == set(stages)
    for name, stage in lock.items():
        assert all(o.get("md5") for o in stage["outs"]), name


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_does_not_ignore_the_files_dvc_needs_committed():
    # snapshots/, not data/: DVC does not look for .dvc files inside git-ignored folders, so a pointer under data/
    # would be skipped by a bare `dvc push`, `dvc pull` and `dvc status`
    for path in (
        "dvc.yaml",
        "dvc.lock",
        ".dvc/config",
        ".dvcignore",
        "metrics/toy_gold.json",
        "snapshots/gold_real.dvc",
        "snapshots/.gitignore",
    ):
        ignored = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0
        assert not ignored, f"{path} is git-ignored"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_the_data_of_the_real_gold_snapshot_stays_git_ignored():
    for path in (
        "snapshots/gold_real/facility_month_dataset.parquet",
        "snapshots/gold_real/_current.json",
    ):
        ignored = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0
        assert ignored, f"{path} must stay git-ignored: only the .dvc pointer is committed"
