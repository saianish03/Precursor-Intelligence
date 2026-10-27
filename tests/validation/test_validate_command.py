"""`precursor-pipeline validate`: exit codes 0 clean, 2 schema violation (or warning with --strict), 3 config."""

import json
import sys

import pytest
import yaml
from validation_fakes import citations

from precursorintelligence.harmonize.cli import main
from precursorintelligence.ingestion.pipeline import EXIT_CONFIG, EXIT_OK, EXIT_VALIDATION

TABLE = "citations_by_vintage"
RUN = "manual__2026-10-26T12.00.00"  # no ':' so the local store can use it in a path on Windows


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["precursor-pipeline", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


@pytest.fixture
def cli(monkeypatch, tmp_path, env_file, cfg, put_silver):
    """Two clean silver months, a validation config file, and the schema inferred through the command."""
    path = tmp_path / "validation.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump()))
    for v in ("2026-07", "2026-08"):
        put_silver(TABLE, v, citations(n=40, vintage=v))
    base = ["--env-file", str(env_file), "--config", str(path), "--run-id", RUN]

    def call(*args):
        return run(monkeypatch, "validate", *args, *base)

    assert call("--layer", "silver", "--write-schema") == EXIT_OK
    return call


def test_help_lists_the_validate_command(monkeypatch, capsys):
    assert run(monkeypatch, "--help") == EXIT_OK
    assert "validate" in capsys.readouterr().out


def test_write_schema_makes_one_yaml_per_table(cli, tmp_path):
    assert (tmp_path / "contracts" / f"{TABLE}.yaml").is_file()


def test_a_clean_layer_exits_0(cli, capsys, store):
    assert cli("--layer", "silver") == EXIT_OK
    out = capsys.readouterr().out
    assert "errors=0 warnings=0" in out and f"reports/validation/run={RUN}/" in out
    assert store.exists(f"reports/validation/run={RUN}/anomalies.json")


def test_a_schema_violation_exits_2_and_is_listed(cli, capsys, put_silver, store):
    put_silver(TABLE, "2026-08", citations(n=40, vintage="2026-08", scope_severity=["Z"] * 40))
    assert cli("--layer", "silver") == EXIT_VALIDATION
    assert (
        "error citations_by_vintage 2026-08 scope_severity allowed_values"
        in capsys.readouterr().err
    )
    with store.open_read(f"reports/validation/run={RUN}/anomalies.json") as f:
        assert json.load(f)["counts"]["error"] == 1


def test_a_warning_exits_0_unless_strict(cli, put_silver):
    put_silver(TABLE, "2026-08", citations(n=20, vintage="2026-08"))
    assert cli("--layer", "silver") == EXIT_OK
    assert cli("--layer", "silver", "--strict") == EXIT_VALIDATION


def test_all_months_and_one_month_run(cli, capsys):
    assert cli("--layer", "silver", "--all-months") == EXIT_OK
    assert "months=2" in capsys.readouterr().out
    assert cli("--layer", "silver", "--month", "2026-07") == EXIT_OK


@pytest.mark.parametrize(
    "args",
    [
        ["--layer", "bronze"],
        ["--layer", "silver", "--table", "ledger_citations"],
        ["--layer", "silver", "--month", "2020-01"],
        ["--layer", "gold"],  # no gold schema file yet
        [],  # --layer is required
    ],
)
def test_config_and_usage_errors_exit_3(cli, args):
    assert cli(*args) == EXIT_CONFIG


def test_the_process_log_is_shipped_next_to_the_reports(cli, store):
    cli("--layer", "silver")
    assert store.list(f"reports/validation/run={RUN}/logs/")


def test_results_are_recorded_in_dq_results(cli, put_silver, ctx):
    put_silver(TABLE, "2026-08", citations(n=40, vintage="2026-08", scope_severity=["Z"] * 40))
    cli("--layer", "silver")
    (row,) = ctx.metadata.read("dq_results", {"run_id": RUN})
    assert (row["check_name"], row["severity"]) == ("allowed_values", "error")
