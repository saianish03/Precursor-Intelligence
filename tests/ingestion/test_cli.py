"""The CLI as a DAG task sees it: exit codes, --run-id, the `entries` JSON line and the `lock` commands.

Metadata is prepared in-process against the fake CMS; the CLI then reads the same store via --env-file,
so no command here touches the network."""

import json
import sys

import pytest
import yaml
from test_pipeline_offline import build_archive

from precursorintelligence.common.io.run_lock import RunLock
from precursorintelligence.ingestion import cli as cli_module
from precursorintelligence.ingestion.cli import main
from precursorintelligence.ingestion.context import build_context
from precursorintelligence.ingestion.http import SafeHttpClient
from precursorintelligence.ingestion.pipeline import (
    EXIT_CONFIG,
    EXIT_ERROR,
    EXIT_OK,
    EXIT_VALIDATION,
    LOCK_NAME,
    run_step1,
)

RUN = "manual__2026-10-13T20:01:02.123456+00:00"


@pytest.fixture
def env_file(tmp_path):
    """An env yaml pointing at the same store and database as make_ctx(root=tmp_path)."""
    path = tmp_path / "env.yaml"
    env = {
        "env": "test-cli",
        "object_store": {"kind": "local", "root": str(tmp_path / "store")},
        "metadata_store": {"url": f"sqlite:///{tmp_path / 'meta.db'}"},
        "logging": {"log_dir": str(tmp_path / "logs")},
    }
    path.write_text(yaml.safe_dump(env))
    return str(path)


def cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["precursor-ingest", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


@pytest.fixture
def downloaded(fake_cms, make_ctx, tmp_path):
    """State after the DAG's discover+download task."""
    build_archive(fake_cms)
    run_step1(make_ctx(fake_cms, root=tmp_path), steps=("discover", "download"))
    return make_ctx(fake_cms, root=tmp_path)


@pytest.fixture
def locked(downloaded):
    """The DAG's first task has run: RUN holds the lock."""
    RunLock(downloaded.metadata, LOCK_NAME).acquire(RUN)
    return downloaded


def test_entries_prints_pending_ids_as_last_json_line(downloaded, env_file, monkeypatch, capsys):
    assert cli(monkeypatch, "entries", "--env-file", env_file, "--pending") == EXIT_OK
    ids = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert len(ids) == 5 and all(i.startswith(("theme:", "current:")) for i in ids)


def test_mapped_task_command_then_nothing_pending(locked, env_file, monkeypatch, capsys):
    cli(monkeypatch, "entries", "--env-file", env_file, "--pending")
    ids = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    for entry_id in ids:
        code = cli(
            monkeypatch,
            "step1",
            "--env-file",
            env_file,
            "--run-id",
            RUN,
            "--steps",
            "extract,inventory",
            "--entry-ids",
            entry_id,
        )
        assert code == EXIT_OK
    cli(monkeypatch, "entries", "--env-file", env_file, "--pending")
    assert capsys.readouterr().out.strip().splitlines()[-1] == "[]"
    logs = locked.object_store.list(f"reports/step1/run={RUN}/logs")
    assert len(logs) == 5, "each task process ships its own log file"


def test_report_task_can_be_retried(locked, env_file, monkeypatch):
    for _ in range(2):
        code = cli(
            monkeypatch,
            "step1",
            "--env-file",
            env_file,
            "--run-id",
            RUN,
            "--steps",
            "resolve,report",
        )
        assert code == EXIT_OK


def test_lock_commands(downloaded, env_file, monkeypatch, capsys):
    assert cli(monkeypatch, "lock", "acquire", "--env-file", env_file, "--run-id", RUN) == 0
    assert RunLock(downloaded.metadata, LOCK_NAME).holder()["run_id"] == RUN
    args = ("step1", "--env-file", env_file, "--steps", "resolve")
    assert cli(monkeypatch, *args, "--run-id", RUN) == EXIT_OK  # a task of the holding run
    assert (
        cli(monkeypatch, *args, "--run-id", "other-run") == EXIT_CONFIG
    )  # its lock task never ran
    assert cli(monkeypatch, *args) == EXIT_ERROR  # a standalone run waits for the DAG run (retry)
    assert RunLock(downloaded.metadata, LOCK_NAME).holder()["run_id"] == RUN
    assert cli(monkeypatch, "lock", "status", "--env-file", env_file) == 0
    assert RUN in capsys.readouterr().out
    assert (
        cli(monkeypatch, "lock", "release", "--env-file", env_file, "--run-id", "x") == EXIT_ERROR
    )
    assert cli(monkeypatch, "lock", "release", "--env-file", env_file, "--run-id", RUN) == 0
    assert RunLock(downloaded.metadata, LOCK_NAME).holder() is None


@pytest.mark.parametrize(
    "args",
    [
        ("step1", "--run-id", "has space"),
        ("step1", "--run-id", "x" * 65),
        ("step1", "--steps", "download,bogus"),
        ("step1", "--zip-dates", "2021-13-01"),
        ("step1", "--entry-ids", ""),
        ("step1", "--entry-ids", " , "),
        ("step1", "--no-such-flag"),
        ("no-such-command",),
        ("step1", "--env", "does-not-exist"),
    ],
)
def test_config_and_usage_errors_exit_3(args, monkeypatch, tmp_path):
    assert cli(monkeypatch, *args) == EXIT_CONFIG
    assert not (tmp_path / "store").exists()


def test_help_exits_0(monkeypatch, capsys):
    assert cli(monkeypatch, "--help") == EXIT_OK
    assert "entries" in capsys.readouterr().out


def test_task_without_its_run_holding_the_lock_is_a_config_error(downloaded, env_file, monkeypatch):
    args = ("step1", "--env-file", env_file, "--run-id", RUN, "--steps", "resolve")
    assert cli(monkeypatch, *args) == EXIT_CONFIG
    assert RunLock(downloaded.metadata, LOCK_NAME).holder() is None, "a task never takes the lock"


@pytest.fixture
def offline(monkeypatch, fake_cms):
    """Route the CLI's HTTP client to the fake CMS (commands that would otherwise call data.cms.gov)."""

    def build(settings, *, run_id=None, http=None):
        s = settings.sources
        fake_http = SafeHttpClient(s.source, s.http, s.security, session=fake_cms)
        return build_context(settings, run_id=run_id, http=fake_http)

    monkeypatch.setattr(cli_module, "build_context", build)
    return fake_cms


def test_catalog_is_read_only_and_ignores_the_lock(locked, offline, env_file, monkeypatch, capsys):
    """It must never store the catalog: check_release would then miss a new or re-sized zip."""
    offline.add("2021-05-27", b"PK" + b"x" * 100)  # newly advertised, not downloaded
    before = sorted(r["entry_id"] for r in locked.metadata.read("catalog"))
    snapshots = locked.object_store.list("landing/cms_nh/catalog")
    assert cli(monkeypatch, "catalog", "--env-file", env_file) == EXIT_OK  # DAG run holds the lock
    assert "nursing-homes_2021-05-27.zip" in capsys.readouterr().out
    assert sorted(r["entry_id"] for r in locked.metadata.read("catalog")) == before
    assert locked.object_store.list("landing/cms_nh/catalog") == snapshots
    assert RunLock(locked.metadata, LOCK_NAME).holder()["run_id"] == RUN


def test_catalog_contract_change_exits_2(offline, env_file, monkeypatch):
    offline.entries.append({"unexpected": "shape"})
    assert cli(monkeypatch, "catalog", "--env-file", env_file) == EXIT_VALIDATION


def test_stale_task_list_exits_3(locked, env_file, monkeypatch):
    args = ("step1", "--env-file", env_file, "--run-id", RUN, "--steps", "extract")
    assert cli(monkeypatch, *args, "--entry-ids", "theme:1999-01-01:gone.zip") == EXIT_CONFIG


def test_lock_acquire_takes_over_a_stale_lock(downloaded, env_file, monkeypatch, capsys):
    downloaded.metadata.upsert(
        "run_lock",
        [{"lock_name": LOCK_NAME, "run_id": "killed", "acquired_at": "2026-01-01T00:00:00+00:00"}],
    )
    assert cli(monkeypatch, "lock", "acquire", "--env-file", env_file, "--run-id", RUN) == EXIT_OK
    held = RunLock(downloaded.metadata, LOCK_NAME).holder()
    assert held["run_id"] == RUN and "took over from stale run killed" in held["holder"]
