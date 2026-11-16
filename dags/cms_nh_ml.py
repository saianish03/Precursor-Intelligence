"""cms_nh_ml: the ML pipeline as one DAG (docs/ML_PIPELINE_SPEC.md §15).

Started by every `cms_nh_gold` event (a published, validated gold) or by hand. Every task except the branches and
`notify` is one `precursor-pipeline ml <command>` in a `precursor-pipeline` container with the run's `--run-id`, so
the run folder is `ml/runs/<run id>/`. No business logic here: `decide` (monitoring/decide.py) picks the path and
prints it; the branches only read it.

    decide ─► go? ─► prepare ─► path ─┬─ RETUNE ─► tune ─────────────────────────────────────────────┐
       │                              ├─ RETRAIN ─► train ─► evaluate ─► gate ─┐                       │
       │                              ├─ PROMOTE ─► promote ───────────────────┤                       │
       │                              └─ SCORE ────────────────────────────────┤                       ▼
       │                                         score ─► scored? ─► publish ─► monitor ──────────► notify
       └─ (SKIP) ───────────────────────────────────────────────────────────────────────────────────────┘

- SKIP stops after `decide` (its `ml_runs` row is written) and `notify` emails decide's summary. REFUSE is exit 2
  and fails the run (alert, no retry); gold not current is exit 1 (retried).
- `score` skips without a champion (first run, awaiting Checkpoint ML-A); then `scored?` skips `publish`, so the
  asset `cms_nh_scores` is emitted only when scores were published. `monitor` and `notify` still run.
- RETUNE (`mode=retune` only, Phase 1) runs `tune` alone: its proposal and plots. `train` doesn't apply a proposal,
  so a re-tune trains, gates and scores nothing (§12.2); a drift flag only asks a human for one.
- `train` and `tune` take the `heavy` pool's one slot (shared with gold's `build_gold`, §15.6).
- Alerts: a failed run emails through on_dag_failure; every finished run emails its summary (`notify`), which
  says when the gate held, a signal flagged or a re-tune was requested (course §7.5).

Created paused: unpause it before the pipeline run whose gold should start it (Airflow keeps no asset events for a
paused DAG).

Params:
  mode              auto | retrain | retune | score_only | promote (§15.4)
  version, reason   promote only: the version to move `champion` to (any earlier one = rollback) and why; passed
                    to the container's environment, never its command line
  allow_unverified  null = ml.yaml's value; true/false overrides it for this run (logged in decision.json)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.sdk import Asset, Param, dag, get_current_context, task
from utils.alerts import on_dag_failure
from utils.pipeline_operator import (
    HEAVY_POOL,
    PipelineDockerOperator,
    docker_kwargs,
    last_json_object,
    pipeline_env,
)

from precursorintelligence.alerts import ml_run_alert, send

GOLD, SCORES = Asset("cms_nh_gold"), Asset("cms_nh_scores")
ML = "precursor-pipeline ml"
DECISION_KEYS = ("decision", "proceed")  # tell the decision from a JSON log line
SUMMARY_KEYS = ("decision", "flags", "retune_requested")  # monitor's summary
TRAIN_TIMEOUT = timedelta(hours=3)  # a hung fit must not hold the heavy pool
MODE = "--mode {{ params.mode }}"
ALLOW = (
    "{{ '' if params.allow_unverified is none else "
    "('--allow-unverified' if params.allow_unverified else '--refuse-unverified') }}"
)
# the branch targets per decision; `score` is on every scoring path so a branch never skips it (its trigger rule
# waits)
PATHS = {
    "RETUNE": ["tune"],
    "RETRAIN": ["train", "score"],
    "PROMOTE": ["promote", "score"],
    "SCORE": ["score"],
}
AFTER_ANY = "none_failed_min_one_success"


def run(task_id: str, command: str, **kwargs) -> PipelineDockerOperator:
    """One `ml` command per task; the env and the DAG run's id are appended to every command."""
    tail = f"--env {pipeline_env()} --run-id {{{{ run_id }}}}"
    return PipelineDockerOperator(
        task_id=task_id, command=f"{ML} {command} {tail}", **{**docker_kwargs(), **kwargs}
    )


def with_output() -> dict:
    """The task's every output line as its XCom: stdout and stderr arrive interleaved."""
    return {"do_xcom_push": True, "xcom_all": True}


@dag(
    dag_id="cms_nh_ml",
    schedule=[GOLD],
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },  # exit 1 only (see the operator)
    on_failure_callback=on_dag_failure,
    params={
        "mode": Param("auto", enum=["auto", "retrain", "retune", "score_only", "promote"]),
        "version": Param(None, type=["null", "integer"], minimum=1),
        "reason": Param("", type="string", description="required for promote"),
        "allow_unverified": Param(None, type=["null", "boolean"]),
    },
    tags=["ml-pipeline"],
    doc_md=__doc__,
)
def cms_nh_ml():
    decide = run("decide", f"decide {MODE} {ALLOW}", **with_output())
    prepare = run("prepare", f"prepare {ALLOW}")  # pins the gold run named in decision.json

    @task.branch(retries=0)
    def go(output: str | list[str] | None) -> str | None:
        return "prepare" if last_json_object(output, DECISION_KEYS)["proceed"] else None

    @task.branch(retries=0)
    def path(output: str | list[str] | None) -> list[str]:
        return PATHS[last_json_object(output, DECISION_KEYS)["decision"]]

    tune = run("tune", "tune", pool=HEAVY_POOL, retries=0, execution_timeout=TRAIN_TIMEOUT)
    train = run(
        "train",
        f"train {MODE}",
        pool=HEAVY_POOL,
        retries=1,
        execution_timeout=TRAIN_TIMEOUT,
    )
    evaluate = run("evaluate", f"evaluate {MODE}")
    gate = run("gate", f"gate {MODE}")
    promote = run(
        "promote",
        "promote",
        environment={
            **docker_kwargs()["environment"],
            "PI_ML_VERSION": "{{ params.version if params.version is not none else '' }}",
            "PI_ML_REASON": "{{ params.reason }}",
        },
    )
    score = run("score", f"score {MODE}", trigger_rule=AFTER_ANY, **with_output())

    # retries=0: reading the summary again cannot change it
    @task.short_circuit(retries=0, ignore_downstream_trigger_rules=False)
    def scored(output: str | list[str] | None) -> bool:
        return not last_json_object(output, ("skipped",))["skipped"]

    publish = run("publish", f"publish {MODE}", outlets=[SCORES])
    monitor = run("monitor", "monitor", trigger_rule=AFTER_ANY, **with_output())

    @task(retries=0, trigger_rule="none_failed")
    def notify() -> None:
        """Email the run's summary: monitor's, or decide's when the run skipped or only re-tuned (§12.3)."""
        context = get_current_context()
        ti = context["ti"]
        output, keys = ti.xcom_pull(task_ids="monitor"), SUMMARY_KEYS
        if output is None:  # SKIP or RETUNE: monitor never ran
            output, keys = ti.xcom_pull(task_ids="decide"), DECISION_KEYS
        send(
            *ml_run_alert(context["dag"].dag_id, context["run_id"], last_json_object(output, keys))
        )

    branch = path(decide.output)
    is_scored = scored(score.output)
    go(decide.output) >> prepare >> branch >> [tune, train, promote, score]
    train >> evaluate >> gate >> score
    promote >> score
    score >> is_scored >> publish
    [publish, is_scored] >> monitor
    [decide, monitor, tune] >> notify()


cms_nh_ml()
