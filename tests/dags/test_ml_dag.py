"""cms_nh_ml (ML spec §15): schedule, params, the ml commands, the branches, trigger rules, the heavy pool and the
notify summary. Runs in the airflow image (conftest.py)."""

import json

import pytest
from airflow.models.dagbag import DagBag
from utils.alerts import on_dag_failure
from utils.pipeline_operator import HEAVY_POOL, PipelineDockerOperator

DAG_ID = "cms_nh_ml"
COMMANDS = {
    "decide": "decide --mode {{ params.mode }}",
    "prepare": "prepare",
    "tune": "tune",
    "train": "train --mode {{ params.mode }}",
    "evaluate": "evaluate --mode {{ params.mode }}",
    "gate": "gate --mode {{ params.mode }}",
    "promote": "promote",
    "score": "score --mode {{ params.mode }}",
    "publish": "publish --mode {{ params.mode }}",
    "monitor": "monitor",
}


@pytest.fixture(scope="module")
def bag(dags_dir):
    mp = pytest.MonkeyPatch()
    mp.setenv("PI_PIPELINE_ENV", "dev")
    bag = DagBag(dag_folder=str(dags_dir), include_examples=False)
    mp.undo()
    assert bag.import_errors == {}
    return bag


@pytest.fixture(scope="module")
def dag(bag):
    return bag.dags[DAG_ID]


def lines(obj):
    return [
        "2026-11-15 INFO starting",
        json.dumps(obj),
        json.dumps({"level": "INFO", "message": "done"}),
    ]


def test_schedule_and_params(dag):
    assert [a.name for _, a in dag.timetable.asset_condition.iter_assets()] == ["cms_nh_gold"]
    assert dag.catchup is False and dag.max_active_runs == 1
    assert dag.is_paused_upon_creation is True
    assert dag.default_args["retries"] == 2
    callbacks = dag.on_failure_callback
    assert on_dag_failure in (callbacks if isinstance(callbacks, list) else [callbacks])
    assert dag.params.get_param("mode").schema["enum"] == [
        "auto",
        "retrain",
        "retune",
        "score_only",
        "promote",
    ]
    assert dag.params["mode"] == "auto" and dag.params["allow_unverified"] is None


@pytest.mark.parametrize("task_id", sorted(COMMANDS))
def test_every_container_task_is_one_ml_command(dag, task_id):
    t = dag.get_task(task_id)
    assert isinstance(t, PipelineDockerOperator)
    assert t.command.startswith(f"precursor-pipeline ml {COMMANDS[task_id]}")
    assert t.command.endswith("--env dev --run-id {{ run_id }}")
    assert t.image == "precursor-pipeline:latest"


def test_wiring_and_trigger_rules(dag):
    up = {t.task_id: t.upstream_task_ids for t in dag.tasks}
    assert up == {
        "decide": set(),
        "go": {"decide"},
        "prepare": {"go"},
        "path": {"decide", "prepare"},
        "tune": {"path"},
        "train": {"path"},
        "evaluate": {"train"},
        "gate": {"evaluate"},
        "promote": {"path"},
        "score": {"path", "gate", "promote"},
        "scored": {"score"},
        "publish": {"scored"},
        "monitor": {"publish", "scored"},
        "notify": {"decide", "monitor", "tune"},
    }
    rules = {t.task_id: t.trigger_rule for t in dag.tasks}
    for task_id in ("score", "monitor"):
        assert rules[task_id] == "none_failed_min_one_success"
    assert rules["notify"] == "none_failed"
    assert [a.name for a in dag.get_task("publish").outlets] == ["cms_nh_scores"]


@pytest.mark.parametrize(("task_id", "minutes"), [("train", 45), ("tune", 90), ("evaluate", 60)])
def test_the_heavy_tasks_share_gold_s_pool_with_a_timeout_and_no_retry(dag, bag, task_id, minutes):
    t = dag.get_task(task_id)
    assert t.pool == HEAVY_POOL == bag.dags["cms_nh_pipeline"].get_task("gold.build_gold").pool
    assert t.retries == 0  # a hung fit alerts instead of holding the slot twice
    assert t.execution_timeout.total_seconds() == minutes * 60


def test_outputs_are_kept_for_the_branches_and_notify(dag):
    for task_id in ("decide", "score", "monitor"):
        t = dag.get_task(task_id)
        assert t.do_xcom_push and t.xcom_all


@pytest.mark.parametrize("task_id", ["decide", "promote"])  # decide checks the request first
def test_promote_gets_version_and_reason_from_the_params_through_the_environment(dag, task_id):
    t = dag.get_task(task_id)
    assert "environment" in t.template_fields
    assert "params.version" in t.environment["PI_ML_VERSION"]
    assert t.environment["PI_ML_REASON"] == "{{ params.reason }}"
    assert "PI_ML" not in t.command


@pytest.mark.parametrize(
    ("decision", "proceed", "expected_go", "expected_path"),
    [
        ("SKIP", False, None, None),
        ("RETRAIN", True, "prepare", ["train", "score"]),
        ("RETUNE", True, "prepare", ["tune"]),  # Phase 1: tune alone, nothing trained or scored
        ("PROMOTE", True, "prepare", ["promote", "score"]),
        ("SCORE", True, "prepare", ["score"]),
    ],
)
def test_the_branches_follow_decide(dag, decision, proceed, expected_go, expected_path):
    out = lines({"decision": decision, "proceed": proceed, "reason": "r"})
    assert dag.get_task("go").python_callable(out) == expected_go
    if expected_path:
        assert dag.get_task("path").python_callable(out) == expected_path


@pytest.mark.parametrize(("skipped", "expected"), [(True, False), (False, True)])
def test_publish_runs_only_when_score_scored(dag, skipped, expected):
    assert dag.get_task("scored").python_callable(lines({"skipped": skipped})) is expected


class FakeTi:
    def __init__(self, outputs):
        self.outputs = outputs

    def xcom_pull(self, task_ids):
        return self.outputs.get(task_ids)


@pytest.mark.parametrize(
    ("outputs", "subject"),
    [
        (
            {
                "monitor": lines(
                    {"decision": "RETRAIN", "v": "2026-10", "flags": [], "retune_requested": False}
                )
            },
            "[cms_nh_ml] RETRAIN 2026-10",
        ),
        (
            {
                "decide": lines(
                    {"decision": "SKIP", "proceed": False, "v": "2026-10", "reason": "nothing new"}
                )
            },
            "[cms_nh_ml] SKIP 2026-10",
        ),
        (
            {
                "decide": lines(
                    {
                        "decision": "RETUNE",
                        "proceed": True,
                        "v": "2026-10",
                        "reason": "re-tune requested",
                    }
                )
            },
            "[cms_nh_ml] RETUNE 2026-10",
        ),
    ],
)
def test_notify_emails_monitor_s_summary_or_decide_s_without_monitor(
    dag, monkeypatch, outputs, subject
):
    fn = dag.get_task("notify").python_callable
    sent = []
    context = {"ti": FakeTi(outputs), "dag": dag, "run_id": "manual__1"}
    monkeypatch.setitem(fn.__globals__, "send", lambda s, b: sent.append((s, b)))
    monkeypatch.setitem(fn.__globals__, "get_current_context", lambda: context)
    fn()
    assert [s for s, _ in sent] == [subject]
