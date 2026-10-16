"""The `precursor-pipeline` entry point: exit codes follow the DAG contract (docs/DATA_PIPELINE_SPEC.md §15.4)."""

import sys

import pytest

from precursorintelligence.harmonize.cli import main
from precursorintelligence.ingestion.pipeline import EXIT_CONFIG, EXIT_OK


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
