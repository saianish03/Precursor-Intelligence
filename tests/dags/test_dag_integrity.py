"""The DAG imports, has no cycles, and wires the spec §15 task contract: every task a PipelineDockerOperator,
one shared --run-id, the lock taken first and released last (a teardown, so a failed run still fails)."""

from datetime import timedelta

import pytest
from airflow.models.dagbag import DagBag
from airflow.task.trigger_rule import TriggerRule
from utils.alerts import on_dag_failure, on_release_failure
from utils.pipeline_operator import PipelineDockerOperator

DAG_ID = "cms_nh_pipeline"
CONTAINER_TASKS = {
    "ingestion.acquire_lock",
    "ingestion.discover_download",
    "ingestion.pending_entries",
    "ingestion.extract_inventory",
    "ingestion.resolve_report",
    "ingestion.check_anomalies",
    "bronze.load_bronze",
    "silver.build_silver",
    "gold.build_gold",
    "release_lock",
}


@pytest.fixture(scope="module")
def dag(dags_dir, monkeypatch_module):
    monkeypatch_module.setenv("PI_PIPELINE_ENV", "dev")
    monkeypatch_module.delenv("PI_DAG_SCHEDULE", raising=False)  # the production default
    bag = DagBag(dag_folder=str(dags_dir), include_examples=False)
    assert bag.import_errors == {}
    return bag.dags[DAG_ID]


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _operator_class(task):
    return getattr(task, "operator_class", type(task))  # a mapped task reports the class it expands


def test_every_container_task_uses_the_pipeline_operator(dag):
    tasks = {t.task_id: t for t in dag.tasks}
    assert set(tasks) >= CONTAINER_TASKS
    for task_id in CONTAINER_TASKS:
        assert issubclass(_operator_class(tasks[task_id]), PipelineDockerOperator), task_id


def test_commands_pass_the_env_and_the_run_id(dag):
    for task in dag.tasks:
        if task.task_id in CONTAINER_TASKS and task.task_id != "ingestion.extract_inventory":
            assert task.command.endswith("--env dev --run-id {{ run_id }}"), task.task_id


def test_node_order_and_lock(dag):
    tasks = {t.task_id: t for t in dag.tasks}

    def upstream(task_id):
        return set(tasks[task_id].upstream_task_ids)

    assert upstream("ingestion.acquire_lock") == set()
    assert upstream("ingestion.discover_download") == {"ingestion.acquire_lock"}
    assert upstream("bronze.load_bronze") == {"continue_after_ingestion"}
    assert upstream("silver.build_silver") == {"continue_after_bronze"}
    assert upstream("continue_after_silver") == {"silver.build_silver"}
    assert upstream("gold.build_gold") == {"continue_after_silver"}
    assert upstream("release_lock") == {
        "gold.build_gold",
        "ingestion.acquire_lock",
        "ingestion.alert_anomalies",
    }
    release = tasks["release_lock"]
    assert release.is_teardown and release.trigger_rule == TriggerRule.ALL_DONE_SETUP_SUCCESS
    assert "ingestion.acquire_lock" in release.upstream_task_ids  # its setup
    assert tasks["ingestion.acquire_lock"].is_setup
    assert release.retries == 2  # owner-checked and idempotent
    # on_failure_fail_dagrun would make the teardown a leaf, and its success would hide a failed run
    assert not release.on_failure_fail_dagrun
    assert tasks["ingestion.resolve_report"].trigger_rule == TriggerRule.NONE_FAILED
    # the run's state comes from its leaves (teardowns aside): gold, and the anomaly alert after resolve
    # (not behind continue_after, so it also runs on a stop_after run). Every other task is upstream of one,
    # so any failure fails the run.
    leaves = {
        t.task_id
        for t in dag.tasks
        if not t.is_teardown and all(tasks[d].is_teardown for d in t.downstream_task_ids)
    }
    assert leaves == {"gold.build_gold", "ingestion.alert_anomalies"}
    assert upstream("ingestion.check_anomalies") == {"ingestion.resolve_report"}
    assert upstream("ingestion.alert_anomalies") == {"ingestion.check_anomalies"}


def test_schedule_and_params(dag):
    assert dag.max_active_runs == 1 and dag.catchup is False
    assert dag.timetable.summary == "0 0 * * *"  # @daily: production polls CMS once a day
    assert set(dag.params) == {"mode", "calibrate", "stop_after", "entry_ids"}
    assert dag.params["mode"] == "monthly" and dag.params["entry_ids"] == ""


def test_alerts_are_wired(dag):
    tasks = {t.task_id: t for t in dag.tasks}
    assert on_dag_failure in _callbacks(dag.on_failure_callback)  # once per failed run
    # a teardown never fails the run, so a failed release alerts on its own
    assert on_release_failure in _callbacks(tasks["release_lock"].on_failure_callback)
    check = tasks["ingestion.check_anomalies"]
    assert " anomalies --severity error --new-only " in check.command and check.do_xcom_push
    # a hung gold run would hold the lock; the timeout fails it, and the DAG callback alerts
    assert tasks["gold.build_gold"].execution_timeout == timedelta(hours=2)


def _callbacks(value):
    return value if isinstance(value, list) else [value]
