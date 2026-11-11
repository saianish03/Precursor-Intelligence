"""cms_nh_release_check: the hourly release check that starts cms_nh_pipeline (docs/DATA_PIPELINE_SPEC.md §2.0 and
§15; decision 24).

`check_release` runs `precursor-pipeline check-release` in a precursor-pipeline container. It reads the CMS metastore
and archive catalog (read-only) and the pipeline's state, records a decision in `vintage_registry` and prints it as
JSON. On READY, RESUME or CHANGED, `emit_release` records an event of the asset `cms_nh_release`, which starts a
cms_nh_pipeline run; the event's `extra` carries the decision, so each pipeline run can be traced to the check that
started it. PARTIAL, BUSY, BLOCKED and NO_CHANGE end here (short-circuit). `alert` emails what the check flags, once
each: a release that stays partial, or proceeds that start no pipeline run. A manual cms_nh_pipeline run never goes
through this DAG, so it always works.

Created paused, and kept paused during development and testing, when cms_nh_pipeline runs only when triggered by
hand. Unpause it when production starts, after cms_nh_pipeline, and only once gold is published; pause it during a
backfill or node-by-node run (docs/runbooks/airflow-local.md §5). A check that fails (CMS unreachable: exit 1) is
retried twice, then alerts (on_failure_callback): one email per failed hourly check.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.sdk import Asset, dag, get_current_context, task
from utils.alerts import on_dag_failure
from utils.pipeline_operator import (
    PipelineDockerOperator,
    docker_kwargs,
    last_json_object,
    pipeline_env,
)

from precursorintelligence.alerts import send

RELEASE = Asset("cms_nh_release")
DECISION_KEYS = ("decision", "proceed")  # tell the decision from a JSON log line
# what the asset event carries to the pipeline run (the full JSON also lists the landing backlog)
EXTRA_KEYS = ("decision", "target", "reason", "latest_ingested", "latest_published", "check_run_id")


@dag(
    dag_id="cms_nh_release_check",
    schedule="@hourly",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },  # exit 1 only (see the operator)
    on_failure_callback=on_dag_failure,  # once, after the last retry
    tags=["data-pipeline", "cms"],
    doc_md=__doc__,
)
def cms_nh_release_check():
    check = PipelineDockerOperator(
        task_id="check_release",
        command=f"precursor-pipeline check-release --env {pipeline_env()} --run-id {{{{ run_id }}}}",
        execution_timeout=timedelta(minutes=20),  # a hung CMS call must not hold max_active_runs
        # every output line: stdout and stderr arrive interleaved, so a JSON log line may follow the decision
        **{**docker_kwargs(), "do_xcom_push": True, "xcom_all": True},
    )

    # retries=0: reading the decision again cannot change it, and an alert is never sent twice
    @task.short_circuit(retries=0)
    def proceed(output: str | list[str] | None) -> bool:
        return bool(last_json_object(output, DECISION_KEYS)["proceed"])

    @task(outlets=[RELEASE])
    def emit_release(output: str | list[str] | None) -> None:
        decision = last_json_object(output, DECISION_KEYS)
        get_current_context()["outlet_events"][RELEASE].extra = {
            k: decision.get(k) for k in EXTRA_KEYS
        }

    @task(retries=0)
    def alert(output: str | list[str] | None) -> None:
        decision = last_json_object(output, DECISION_KEYS)
        if decision.get("alert"):
            send(decision["alert_subject"], decision["alert"])

    proceed(check.output) >> emit_release(check.output)
    alert(check.output)


cms_nh_release_check()
