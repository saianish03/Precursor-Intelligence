"""`anomalies --new-only`: the input of the DAG's anomaly alert. resolve re-records an unfixed problem on every
run, so only anomalies the previous successful resolve did not have are listed (one alert per problem)."""

import json

import pytest
from test_cli import cli, env_file  # noqa: F401  (fixture)

from precursorintelligence.ingestion.anomalies import NoResolveError, run_anomalies
from precursorintelligence.ingestion.pipeline import EXIT_CONFIG, EXIT_OK


def _resolve(ctx, run_id, started, status="ok", anomalies=()):
    ctx.metadata.append(
        "run_log", [{"run_id": run_id, "task": "resolve", "status": status, "started_at": started}]
    )
    ctx.metadata.append(
        "anomalies",
        [
            {"run_id": run_id, "entry_id": e, "kind": k, "severity": s, "detail": f"{k} in {e}"}
            for e, k, s in anomalies
        ],
    )


NO_VINTAGE = ("theme:2026-09-28:nh.zip", "no_vintage", "error")
PARTIAL = ("theme:2026-09-28:nh.zip", "partial_zip", "warning")
FAILED_DL = ("theme:2026-10-08:nh.zip", "download_failed", "error")


@pytest.fixture
def ctx(fake_cms, make_ctx, tmp_path):
    return make_ctx(fake_cms, root=tmp_path)


def _kinds(rows):
    return [(a["entry_id"], a["kind"]) for a in rows]


def test_first_run_lists_its_error_anomalies(ctx):
    _resolve(ctx, "r1", "2026-10-01T00:00:00", anomalies=[NO_VINTAGE, PARTIAL])
    assert _kinds(run_anomalies(ctx.metadata, "r1", new_only=True)) == [NO_VINTAGE[:2]]
    assert len(run_anomalies(ctx.metadata, "r1", "warning")) == 2


def test_a_repeated_anomaly_is_not_new(ctx):
    _resolve(ctx, "r1", "2026-10-01T00:00:00", anomalies=[NO_VINTAGE])
    _resolve(ctx, "r2", "2026-10-02T00:00:00", anomalies=[NO_VINTAGE, FAILED_DL])
    assert _kinds(run_anomalies(ctx.metadata, "r2", new_only=True)) == [FAILED_DL[:2]]
    assert len(run_anomalies(ctx.metadata, "r2")) == 2  # without --new-only: all of them


def test_a_failed_resolve_is_not_the_baseline(ctx):
    """A failed resolve failed its DAG run (that alert names the task, not each anomaly), so what it found
    is still reported by the next successful run."""
    _resolve(ctx, "r1", "2026-10-01T00:00:00", anomalies=[])
    _resolve(ctx, "r2", "2026-10-02T00:00:00", status="failed", anomalies=[NO_VINTAGE])
    _resolve(ctx, "r3", "2026-10-03T00:00:00", anomalies=[NO_VINTAGE])
    assert _kinds(run_anomalies(ctx.metadata, "r3", new_only=True)) == [NO_VINTAGE[:2]]


def test_a_fixed_then_returning_problem_is_new_again(ctx):
    _resolve(ctx, "r1", "2026-10-01T00:00:00", anomalies=[NO_VINTAGE])
    _resolve(ctx, "r2", "2026-10-02T00:00:00", anomalies=[])
    _resolve(ctx, "r3", "2026-10-03T00:00:00", anomalies=[NO_VINTAGE])
    assert _kinds(run_anomalies(ctx.metadata, "r3", new_only=True)) == [NO_VINTAGE[:2]]


def test_a_retried_resolve_counts_once(ctx):
    _resolve(ctx, "r1", "2026-10-01T00:00:00", anomalies=[NO_VINTAGE])
    _resolve(ctx, "r2", "2026-10-02T00:00:00", status="failed")
    _resolve(ctx, "r2", "2026-10-02T00:10:00")  # its retry; r2 stays the run under test
    assert run_anomalies(ctx.metadata, "r2", new_only=True) == []


def test_a_run_without_resolve_is_a_wiring_error(ctx):
    with pytest.raises(NoResolveError):
        run_anomalies(ctx.metadata, "r9", new_only=True)


def test_cli_prints_new_error_anomalies_as_last_json_line(ctx, env_file, monkeypatch, capsys):  # noqa: F811
    _resolve(ctx, "r1", "2026-10-01T00:00:00", anomalies=[NO_VINTAGE])
    _resolve(ctx, "r2", "2026-10-02T00:00:00", anomalies=[NO_VINTAGE, FAILED_DL, PARTIAL])
    args = ("anomalies", "--env-file", env_file, "--run-id", "r2", "--severity", "error")
    assert cli(monkeypatch, *args, "--new-only") == EXIT_OK
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == [
        "download_failed (theme:2026-10-08:nh.zip): download_failed in theme:2026-10-08:nh.zip"
    ]
    assert cli(monkeypatch, *args) == EXIT_OK
    assert len(json.loads(capsys.readouterr().out.strip().splitlines()[-1])) == 2


@pytest.mark.parametrize(
    "args",
    [("--run-id", "r1", "--severity", "fatal"), ("--run-id", "never-resolved", "--new-only")],
)
def test_cli_config_errors_exit_3(ctx, env_file, monkeypatch, args):  # noqa: F811
    assert cli(monkeypatch, "anomalies", "--env-file", env_file, *args) == EXIT_CONFIG
