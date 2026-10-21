"""cms_nh_pipeline: the CMS nursing-home data pipeline as one DAG (docs/DATA_PIPELINE_SPEC.md §15).

Nodes are TaskGroups; every task is one CLI command in a `precursor-pipeline` container, and every task of a
run passes the same `--run-id {{ run_id }}`. The `data_pipeline` run lock is taken by the first ingestion task
and released by the last task of the DAG (a teardown, so it always runs once the lock was taken but never
hides a failed run), so all nodes run under one lock. No business logic here.

Implemented nodes: ingestion → bronze → silver. Not built yet, slots marked TODO: `check_release` (spec §2.0,
gates the DAG) and `gold` (spec §7-§11).

Params:
  mode        monthly | backfill (backfill implies the silver calibration tasks)
  calibrate   run the silver calibration tasks (Checkpoints 3-4 proposals) in a monthly run
  stop_after  ingestion | bronze | silver: skip the nodes after it (first backfill, node by node)
  entry_ids   development only: comma-separated catalog entry ids to ingest; empty = every in-scope zip
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

from airflow.sdk import Asset, Param, dag, get_current_context, task, task_group
from utils.pipeline_operator import (
    PipelineDockerOperator,
    docker_kwargs,
    last_json_list,
    pipeline_env,
)

LANDING, BRONZE, SILVER = (
    Asset(name) for name in ("cms_nh_landing", "cms_nh_bronze", "cms_nh_silver")
)
INGEST, PIPELINE = "precursor-ingest", "precursor-pipeline"
NODES = ("ingestion", "bronze", "silver")
ENTRY_ID = re.compile(r"^[A-Za-z0-9_.:\-]+$")  # e.g. theme:2026-08-26:nursing-homes_2026-08-26.zip


def _tail() -> str:
    return f"--env {pipeline_env()} --run-id {{{{ run_id }}}}"


def run(task_id: str, command: str, **kwargs) -> PipelineDockerOperator:
    """One CLI command per task; the env and the DAG run's id are appended to every command."""
    return PipelineDockerOperator(
        task_id=task_id, command=f"{command} {_tail()}", **{**docker_kwargs(), **kwargs}
    )


@dag(
    dag_id="cms_nh_pipeline",
    # CMS sends no notification, so production polls daily (TODO: check_release decides whether a day proceeds).
    # PI_DAG_SCHEDULE="" (laptop) = manual runs only, so unpausing never starts a full-archive run.
    schedule=os.environ.get("PI_DAG_SCHEDULE", "@daily") or None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },  # exit 1 only (see the operator)
    params={
        "mode": Param("monthly", enum=["monthly", "backfill"]),
        "calibrate": Param(False, type="boolean"),
        "stop_after": Param(None, type=["null", "string"], enum=[None, *NODES]),
        "entry_ids": Param(
            "",
            type="string",
            pattern=r"^[A-Za-z0-9_.:,\-]*$",  # catalog entry ids only: nothing that adds CLI arguments
            description="development only: comma-separated catalog entry ids; empty = every in-scope zip",
        ),
    },
    tags=["data-pipeline", "cms"],
    doc_md=__doc__,
)
def cms_nh_pipeline():
    # TODO(check_release): spec §2.0 / §15.5: gate (@task.branch on params.run_check_release) →
    # check_release (`precursor-pipeline check-release`) → @task.short_circuit on its JSON `proceed`.

    # stop_after: skip the next node only; the later nodes skip through their default trigger rule, and the
    # release_lock teardown still runs.
    @task.short_circuit(ignore_downstream_trigger_rules=False)
    def continue_after(node: str) -> bool:
        return get_current_context()["params"]["stop_after"] != node

    @task
    def mapped_commands(output: str | list[str] | None) -> list[str]:
        """`entries --pending` prints the pending entry ids as a JSON list; one extract+inventory command each."""
        run_id = get_current_context()["run_id"]
        entry_ids = last_json_list(output)
        if bad := [e for e in entry_ids if not ENTRY_ID.match(e)]:
            raise ValueError(f"unexpected entry ids from `entries --pending`: {bad}")
        return [
            f"{INGEST} step1 --steps extract,inventory --entry-ids {entry_id} "
            f"--env {pipeline_env()} --run-id {run_id}"
            for entry_id in entry_ids
        ]

    @task_group(group_id="ingestion")
    def ingestion():
        acquire = run("acquire_lock", f"{INGEST} lock acquire")
        fetch = run(
            "discover_download",
            f"{INGEST} step1 --steps discover,download"
            "{% if params.entry_ids %} --entry-ids {{ params.entry_ids }}{% endif %}",
        )
        pending = run(
            "pending_entries", f"{INGEST} entries --pending", do_xcom_push=True, xcom_all=True
        )
        per_zip = PipelineDockerOperator.partial(
            task_id="extract_inventory", max_active_tis_per_dagrun=3, **docker_kwargs()
        ).expand(command=mapped_commands(pending.output))
        resolve = run(
            "resolve_report",
            f"{INGEST} step1 --steps resolve,report",
            trigger_rule="none_failed",  # also runs when no zip was pending (mapped task skipped)
            outlets=[LANDING],
        )
        acquire >> fetch >> pending
        per_zip >> resolve
        return acquire, resolve

    @task_group(group_id="bronze")
    def bronze():
        return run("load_bronze", f"{PIPELINE} bronze", outlets=[BRONZE])

    @task_group(group_id="silver")
    def silver():
        return run(
            "build_silver",
            f"{PIPELINE} silver"
            " {{ '--calibrate' if params.calibrate or params.mode == 'backfill' else '--no-calibrate' }}",
            outlets=[SILVER],
        )

    # TODO(gold): spec §7-§11: gold TaskGroup (`precursor-pipeline gold`, gate, publish; emits cms_nh_gold)
    # between silver and release_lock; later `mirror_metadata` (spec §15.3).

    # A teardown of acquire_lock: it runs whenever the lock was taken (also after a failure or a stop_after
    # skip) but does not decide the DAG run's state. A plain all_done leaf would make every failed run show
    # `success`. If acquire_lock fails (another run holds the lock), nothing is released.
    # Owner-checked and idempotent, so it may retry. Not on_failure_fail_dagrun: that makes Airflow count the
    # teardown as a leaf, and then its success would hide a failed run. A release that still fails shows as a
    # failed task; the next run's acquire_lock then fails (exit 1) or takes over the lock after 12 h.
    release = run(
        "release_lock", f"{INGEST} lock release", retries=2, retry_delay=timedelta(seconds=30)
    )

    acquire, landed = ingestion()
    bronze_done, silver_done = bronze(), silver()
    (
        landed
        >> continue_after.override(task_id="continue_after_ingestion")("ingestion")
        >> bronze_done
    )
    bronze_done >> continue_after.override(task_id="continue_after_bronze")("bronze") >> silver_done
    silver_done >> release.as_teardown(setups=acquire)


cms_nh_pipeline()
