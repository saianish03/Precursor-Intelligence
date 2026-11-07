"""cms_nh_release_check (spec §2.0, §15): hourly, created paused; check_release, then a short-circuit on the decision,
then the asset event that starts cms_nh_pipeline, carrying the decision. Runs in the airflow image (conftest.py)."""

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
from airflow.models.dagbag import DagBag
from utils.alerts import on_dag_failure
from utils.pipeline_operator import PipelineDockerOperator, last_json_object

DAG_ID = "cms_nh_release_check"
DECISION = {
    "decision": "READY",
    "proceed": True,
    "target": "2026-10",
    "reason": "all 7 datasets at 2026-10",
    "alert": None,
    "alert_subject": None,
    "latest_ingested": "2026-09",
    "latest_published": "2026-09",
    "backlog": [],
    "check_run_id": "scheduled__2026-11-07T12:00:00+00:00",
}


@pytest.fixture(scope="module")
def dag(dags_dir):
    mp = pytest.MonkeyPatch()
    mp.setenv("PI_PIPELINE_ENV", "dev")
    bag = DagBag(dag_folder=str(dags_dir), include_examples=False)
    mp.undo()
    assert bag.import_errors == {}
    return bag.dags[DAG_ID]


def output(decision):
    """What the operator returns with xcom_all: every line, log lines interleaved; in prod the log lines are
    JSON too, and one may follow the decision."""
    log = {"level": "INFO", "message": "check_release: done", "run_id": "r"}
    return ["2026-11-07 INFO fetching the catalog", json.dumps(decision), json.dumps(log)]


def test_schedule(dag):
    assert dag.timetable.summary == "0 * * * *"  # hourly: CMS sends no notification
    assert dag.catchup is False and dag.max_active_runs == 1
    assert dag.is_paused_upon_creation is True  # stays paused during development (decision 24)
    assert dag.default_args["retries"] == 2  # exit 1 (CMS unreachable) is retried
    callbacks = dag.on_failure_callback
    assert on_dag_failure in (callbacks if isinstance(callbacks, list) else [callbacks])


def test_check_release_is_one_pipeline_command_whose_output_is_kept(dag):
    check = dag.get_task("check_release")
    assert isinstance(check, PipelineDockerOperator)
    assert check.command == "precursor-pipeline check-release --env dev --run-id {{ run_id }}"
    assert check.do_xcom_push and check.xcom_all
    assert check.execution_timeout == timedelta(minutes=20)
    # a decision that can't be read fails at once, and an alert is never sent twice
    assert dag.get_task("proceed").retries == 0 and dag.get_task("alert").retries == 0


def test_wiring(dag):
    upstream = {t.task_id: t.upstream_task_ids for t in dag.tasks}
    assert upstream == {
        "check_release": set(),
        "proceed": {"check_release"},
        "emit_release": {"check_release", "proceed"},
        "alert": {"check_release"},  # not behind the short-circuit: a PARTIAL alert never proceeds
    }
    assert [a.name for a in dag.get_task("emit_release").outlets] == ["cms_nh_release"]


@pytest.mark.parametrize(("proceed", "expected"), [(True, True), (False, False)])
def test_proceed_follows_the_decision(dag, proceed, expected):
    fn = dag.get_task("proceed").python_callable
    assert fn(output(DECISION | {"proceed": proceed})) is expected


def test_emit_release_puts_the_decision_on_the_asset_event(dag, monkeypatch):
    fn = dag.get_task("emit_release").python_callable
    event = SimpleNamespace(extra=None)

    class OutletEvents:  # like Airflow's accessor: indexed by asset (an Asset is not hashable)
        def __getitem__(self, asset):
            assert asset.name == "cms_nh_release"
            return event

    monkeypatch.setitem(
        fn.__globals__, "get_current_context", lambda: {"outlet_events": OutletEvents()}
    )
    fn(output(DECISION))
    assert event.extra == {
        "decision": "READY",
        "target": "2026-10",
        "reason": "all 7 datasets at 2026-10",
        "latest_ingested": "2026-09",
        "latest_published": "2026-09",
        "check_run_id": "scheduled__2026-11-07T12:00:00+00:00",
    }


def test_alert_emails_only_when_the_check_says_so(dag, monkeypatch):
    fn = dag.get_task("alert").python_callable
    sent = []
    monkeypatch.setitem(fn.__globals__, "send", lambda subject, body: sent.append((subject, body)))
    fn(output(DECISION | {"decision": "PARTIAL", "proceed": False, "alert": None}))
    assert sent == []
    subject = "[cms_nh_release_check] CMS release still partial (PARTIAL 2026-10)"
    fn(
        output(
            DECISION
            | {"decision": "PARTIAL", "alert": "partial for 3 days", "alert_subject": subject}
        )
    )
    assert sent == [(subject, "partial for 3 days")]


def test_last_json_object_takes_the_last_object_with_the_keys_and_refuses_none():
    keys = ("decision", "proceed")
    first, last = {"decision": "READY", "proceed": True}, {"decision": "BUSY", "proceed": False}
    lines = [json.dumps(first), json.dumps(last), '{"level": "INFO"}', '["a list"]', "log line"]
    assert last_json_object(lines, keys) == last
    assert last_json_object("\n".join(lines), keys) == last
    with pytest.raises(ValueError, match="no JSON object"):
        last_json_object(["log line", '{"level": "INFO"}', '["only a list"]'], keys)
