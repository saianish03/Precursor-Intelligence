"""Airflow callbacks for the pipeline alerts: the DAG's on_failure_callback and release_lock's. Building and
sending the emails lives in precursorintelligence.alerts (src/, mounted read-only into the Airflow
containers); this module only reads the run from Airflow's context. Callbacks run inside Airflow's own
containers (the DAG callback in the dag-processor, task callbacks in the LocalExecutor worker), which reach
the internet.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

from utils.pipeline_operator import EXIT_CODE_XCOM

from precursorintelligence.alerts import dag_failure_alert, send

log = logging.getLogger(__name__)


def _base_url() -> str:
    # lazy: keeps this module importable without a configured Airflow
    from airflow.configuration import conf

    return conf.get("api", "base_url", fallback="http://localhost:8080/").rstrip("/")


def log_url(dag_id: str, run_id: str, task_id: str, map_index: int | None = None) -> str:
    mapped = f"/mapped/{map_index}" if map_index is not None and map_index >= 0 else ""
    return f"{_base_url()}/dags/{dag_id}/runs/{quote(run_id, safe='')}/tasks/{task_id}{mapped}"


def _failed_tasks(context: Any) -> list[tuple[str, int | None]]:
    """(task_id, map_index) of the run's failed tasks. The states come keyed by task id, a mapped task's
    as `<task_id>_<map_index>`."""
    ti, dag = context.get("ti"), context["dag"]
    if ti is None:  # Airflow gives no task instance when no task ran
        return []
    states = ti.get_task_states(dag_id=dag.dag_id, run_ids=[context["run_id"]])
    task_ids = set(dag.task_ids)
    out: list[tuple[str, int | None]] = []
    for key, state in sorted(states.get(context["run_id"], {}).items()):
        if state != "failed":
            continue
        if key in task_ids:
            out.append((key, None))
        else:
            task_id, _, index = key.rpartition("_")
            out.append((task_id, int(index)))
    return out


def on_dag_failure(context: Any) -> None:
    """DAG on_failure_callback: Airflow calls it once, after the last retry, so retries do not each alert."""
    dag_id, run_id = context["dag"].dag_id, context["run_id"]
    failed: list[tuple[str, int | None, int | None, str]] = []
    try:
        for task_id, map_index in _failed_tasks(context):
            mapped = {} if map_index is None else {"map_indexes": map_index}
            code = context["ti"].xcom_pull(task_ids=task_id, key=EXIT_CODE_XCOM, **mapped)
            link = log_url(dag_id, run_id, task_id, map_index)
            failed.append((task_id, map_index, code if isinstance(code, int) else None, link))
    except Exception as exc:  # noqa: BLE001 (still alert, without the task list)
        log.warning("could not list the failed tasks: %s", exc)
    send(*dag_failure_alert(dag_id, run_id, failed, context.get("reason")))


def on_release_failure(context: Any) -> None:
    """release_lock's on_failure_callback. A teardown never decides the run's state, so this run may show
    success; the next run's acquire_lock then fails until the lock is freed, or takes it over after 12 h."""
    ti = context["ti"]
    exc = context.get("exception")
    send(
        f"[{ti.dag_id}] {ti.task_id} failed (the run may still show success)",
        "\n".join(
            [
                f"DAG {ti.dag_id}: {ti.task_id} failed in run {ti.run_id}.",
                "A teardown does not decide the run's state, so the run may still show success.",
                f"Error: {str(exc)[:300] if exc else 'unknown'}",
                "",
                "The next run cannot take the lock until it is freed:",
                f"  precursor-ingest lock release --run-id '{ti.run_id}'",
                f"log: {log_url(ti.dag_id, ti.run_id, ti.task_id)}",
            ]
        ),
    )
