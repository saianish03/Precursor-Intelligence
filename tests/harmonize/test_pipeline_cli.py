"""The `precursor-pipeline` entry point: exit codes follow the DAG contract (docs/DATA_PIPELINE_SPEC.md §15.4)."""

import json
import sys

import pytest
import yaml
from bronze_fakes import csv_text

from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore
from precursorintelligence.common.io.object_store import LocalObjectStore
from precursorintelligence.common.io.run_lock import RunLock
from precursorintelligence.harmonize.cli import main
from precursorintelligence.harmonize.reader import decode_csv, parse_csv_text
from precursorintelligence.ingestion.layout import header_hash, normalize_header
from precursorintelligence.ingestion.pipeline import (
    EXIT_CONFIG,
    EXIT_ERROR,
    EXIT_OK,
    EXIT_VALIDATION,
    LOCK_NAME,
)


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["precursor-pipeline", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


def test_help_lists_the_bronze_command(monkeypatch, capsys):
    assert run(monkeypatch, "--help") == EXIT_OK
    assert "bronze" in capsys.readouterr().out


def test_invalid_run_id_is_a_config_error(monkeypatch, capsys):
    assert run(monkeypatch, "bronze", "--run-id", "not a valid id!") == EXIT_CONFIG
    assert "invalid run id" in capsys.readouterr().err


def test_usage_error_maps_to_config_code_not_data_failure_code(monkeypatch):
    assert run(monkeypatch, "bronze", "--no-such-flag") == EXIT_CONFIG


# --- the bronze command against a real local store and metadata DB (read via --env-file) ---------------


RUN = "manual__2026-10-16T12:00:00+00:00"


@pytest.fixture
def env(tmp_path):
    """An env yaml plus handles on the store and database it points at."""
    path = tmp_path / "env.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "env": "test-bronze-cli",
                "object_store": {"kind": "local", "root": str(tmp_path / "store")},
                "metadata_store": {"url": f"sqlite:///{tmp_path / 'meta.db'}"},
                "logging": {"log_dir": str(tmp_path / "logs")},
            }
        )
    )
    store = LocalObjectStore(tmp_path / "store")
    meta = SqlAlchemyMetadataStore(f"sqlite:///{tmp_path / 'meta.db'}")
    meta.ensure_schema()

    class Env:
        file = str(path)

        def stage(
            self, table="provider_info", vintage="2025-06", status="selected", row_count=None
        ):
            text = csv_text([("015009", "Oak Manor"), ("15E123", "Pine Court")])
            parsed = parse_csv_text(decode_csv(text.encode())[0])
            key = f"landing/cms_nh/extracted/{table}_{vintage}.csv"
            store.put_bytes(key, text.encode(), overwrite=True)
            meta.append(
                "inventory",
                [
                    {
                        "entry_id": f"current:{vintage}:{table}",
                        "member_path": f"{table}.csv",
                        "zip_name": "nh.zip",
                        "zip_sha256": "a" * 64,
                        "table_name": table,
                        "extracted_key": key,
                        "processing_date": f"{vintage}-01",
                        "vintage_month": vintage,
                        "row_count": parsed.frame.height if row_count is None else row_count,
                        "header_hash": header_hash(normalize_header(parsed.frame.columns)),
                        "status": status,
                    }
                ],
            )

    e = Env()
    e.store, e.meta = store, meta
    return e


def bronze(monkeypatch, env, *extra):
    return run(monkeypatch, "bronze", "--env-file", env.file, *extra)


def test_bronze_loads_the_selected_files_and_prints_a_summary(env, monkeypatch, capsys):
    env.stage()
    assert bronze(monkeypatch, env) == EXIT_OK
    assert env.store.exists("bronze/provider_info/vintage=2025-06/part-0.parquet")
    assert "loaded=1" in capsys.readouterr().out


def test_a_second_run_changes_nothing_and_still_succeeds(env, monkeypatch, capsys):
    env.stage()
    bronze(monkeypatch, env)
    capsys.readouterr()
    assert bronze(monkeypatch, env) == EXIT_OK
    assert "skipped=1" in capsys.readouterr().out
    assert len(env.meta.read("bronze_files")) == 1


def test_a_file_that_fails_reconciliation_exits_2_and_names_the_file(env, monkeypatch, capsys):
    env.stage()
    env.stage(table="penalties", row_count=99)
    assert bronze(monkeypatch, env) == EXIT_VALIDATION
    err = capsys.readouterr().err
    assert "penalties 2025-06" in err and "row count" in err
    assert env.store.exists("bronze/provider_info/vintage=2025-06/part-0.parquet")


def test_a_pending_inventory_row_exits_2(env, monkeypatch, capsys):
    env.stage()
    env.stage(table="penalties", status="pending")
    assert bronze(monkeypatch, env) == EXIT_VALIDATION
    assert "pending" in capsys.readouterr().err


def test_the_manifest_path_is_printed_and_the_file_exists(env, monkeypatch, capsys):
    env.stage()
    bronze(monkeypatch, env)
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "manifest:" in ln)
    key = line.split("manifest:", 1)[1].strip()
    assert json.loads(env.store.open_read(key).read())["exit_code"] == EXIT_OK


def test_a_dag_task_needs_its_run_to_hold_the_lock(env, monkeypatch):
    env.stage()
    assert bronze(monkeypatch, env, "--run-id", RUN) == EXIT_CONFIG  # the lock task never ran
    RunLock(env.meta, LOCK_NAME).acquire(RUN)
    assert bronze(monkeypatch, env, "--run-id", RUN) == EXIT_OK
    assert (
        RunLock(env.meta, LOCK_NAME).holder()["run_id"] == RUN
    )  # the last task releases it, not this one


def test_a_standalone_run_while_another_run_holds_the_lock_is_a_retryable_error(env, monkeypatch):
    env.stage()
    RunLock(env.meta, LOCK_NAME).acquire("someone-else")
    assert bronze(monkeypatch, env) == EXIT_ERROR
    assert env.meta.read("bronze_files") == []


def test_the_process_log_is_shipped_next_to_the_run_reports(env, monkeypatch):
    env.stage()
    RunLock(env.meta, LOCK_NAME).acquire(RUN)
    bronze(monkeypatch, env, "--run-id", RUN)
    assert len(env.store.list(f"reports/bronze/run={RUN}/logs")) == 1


def test_verify_prints_the_report_and_changes_nothing(env, monkeypatch, capsys):
    env.stage()
    bronze(monkeypatch, env)
    capsys.readouterr()
    objects, rows = env.store.list(""), env.meta.read("bronze_files")
    RunLock(env.meta, LOCK_NAME).acquire("another-run")  # verify takes no lock

    assert bronze(monkeypatch, env, "--verify") == EXIT_OK
    out = capsys.readouterr().out
    assert "valid partitions          1 / 1" in out
    assert env.store.list("") == objects and env.meta.read("bronze_files") == rows


def test_verify_exits_2_when_a_selected_file_has_no_partition(env, monkeypatch, capsys):
    env.stage()  # selected in the inventory, never loaded
    assert bronze(monkeypatch, env, "--verify") == EXIT_VALIDATION
    assert "provider_info 2025-06" in capsys.readouterr().out


def test_a_failed_every_file_check_exits_2_even_when_every_file_loaded(env, monkeypatch, capsys):
    env.stage()
    monkeypatch.setattr(
        "precursorintelligence.harmonize.pipeline.files_without_valid_partition",
        lambda ctx, selected: ["provider_info 2025-06"],
    )
    assert bronze(monkeypatch, env) == EXIT_VALIDATION
    assert "provider_info 2025-06" in capsys.readouterr().err
