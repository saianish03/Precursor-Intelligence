"""The DAG imports, has no cycles, and wires the spec §15 task contract: every task a PipelineDockerOperator,
one shared --run-id, the lock taken first and released last (a teardown, so a failed run still fails)."""

import pytest
from airflow.models.dagbag import DagBag
from airflow.task.trigger_rule import TriggerRule
from utils.pipeline_operator import PipelineDockerOperator

DAG_ID = "cms_nh_pipeline"
CONTAINER_TASKS = {
    "ingestion.acquire_lock",
    "ingestion.discover_download",
    "ingestion.pending_entries",
    "ingestion.extract_inventory",
    "ingestion.resolve_report",
    "bronze.load_bronze",
    "silver.build_silver",
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
    assert upstream("release_lock") == {"silver.build_silver", "ingestion.acquire_lock"}
    release = tasks["release_lock"]
    assert release.is_teardown and release.trigger_rule == TriggerRule.ALL_DONE_SETUP_SUCCESS
    assert "ingestion.acquire_lock" in release.upstream_task_ids  # its setup
    assert tasks["ingestion.acquire_lock"].is_setup
    assert release.retries == 2  # owner-checked and idempotent
    # on_failure_fail_dagrun would make the teardown a leaf, and its success would hide a failed run
    assert not release.on_failure_fail_dagrun
    assert tasks["ingestion.resolve_report"].trigger_rule == TriggerRule.NONE_FAILED


def test_schedule_and_params(dag):
    assert dag.max_active_runs == 1 and dag.catchup is False
    assert dag.timetable.summary == "0 0 * * *"  # @daily: production polls CMS once a day
    assert set(dag.params) == {"mode", "calibrate", "stop_after", "entry_ids"}
    assert dag.params["mode"] == "monthly" and dag.params["entry_ids"] == ""
