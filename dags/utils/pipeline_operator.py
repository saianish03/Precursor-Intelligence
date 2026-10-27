"""How every data-pipeline task runs: one `precursor-pipeline` container per task (spec §15).

`PipelineDockerOperator` adds the pipeline's exit-code contract to the DockerOperator: exit 1 is an error that
a retry may fix; exit 2 (bad or changed data) and exit 3 (config or wiring) fail at once, without retries
(decision §14 #21), and so does 137 (killed, usually out of memory: a retry rarely helps). A failed container's
exit code is kept as an XCom for the failure alert (utils/alerts.py). `docker_kwargs()` reads the deployment from the environment, so the same DAG runs on a laptop
(local disk bind-mounted as the bucket) and on the GCP VM (GCS, no mount). Kept free of heavy imports: the DAG
processor imports this on every parse.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from airflow.exceptions import AirflowFailException
from airflow.providers.docker.exceptions import DockerContainerFailedException
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

NO_RETRY_EXIT_CODES = (2, 3, 137)
EXIT_KINDS = {
    1: "error",
    2: "data or validation failure",
    3: "config or usage error",
    137: "killed, usually out of memory",
}
EXIT_CODE_XCOM = "pipeline_exit_code"
# apache-airflow-providers-docker 4.6.1 raises DockerContainerFailedException(f"Docker container failed:
# {result!r}") with result = docker's wait() dict, e.g. {'Error': None, 'StatusCode': 2}. Checked against that
# release's source; tests/dags/test_pipeline_operator.py fails if a provider upgrade changes it.
_STATUS_CODE = re.compile(r"'StatusCode': (\d+)")


def exit_code_of(exc: DockerContainerFailedException) -> int | None:
    match = _STATUS_CODE.search(str(exc))
    return int(match.group(1)) if match else None


class PipelineDockerOperator(DockerOperator):
    """A DockerOperator whose exit 2, 3 and 137 are not retried."""

    def execute(self, context: Any) -> Any:
        try:
            return super().execute(context)
        except DockerContainerFailedException as exc:
            code = exit_code_of(exc)
            if code is not None and (ti := context.get("ti")) is not None:
                ti.xcom_push(key=EXIT_CODE_XCOM, value=code)
            if code in NO_RETRY_EXIT_CODES:
                raise AirflowFailException(
                    f"pipeline exit {code} ({EXIT_KINDS[code]}), not retried: {exc}"
                ) from exc
            raise


def last_json_list(output: str | list[str] | None) -> list[str]:
    """The JSON list a command printed (e.g. `entries --pending`). The operator returns every output line
    (xcom_all) because stdout and stderr arrive interleaved, so a log line may follow the JSON; take the
    last line that is a JSON list of strings. Nothing printed means an empty list."""
    lines = output if isinstance(output, list) else (output or "").splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return value
    return []


def pipeline_env() -> str:
    """`dev` = local object store under /app/data; `prod` = GCS (PI_BUCKET_URI). Both use PI_DATABASE_URL."""
    return os.environ.get("PI_PIPELINE_ENV", "prod")


def docker_kwargs() -> dict[str, Any]:
    """Container settings shared by every task, from the Airflow deployment's environment (.env)."""
    env = os.environ
    environment = {
        "PYTHONUNBUFFERED": "1"
    }  # results are read from the output lines: never buffer them
    if env.get("PI_BUCKET_URI"):  # unset, not empty: prod.yaml falls back to its default bucket
        environment["PI_BUCKET_URI"] = env["PI_BUCKET_URI"]
    mounts = []
    if env.get("PI_HOST_DATA_DIR"):  # local only: a host folder plays the bucket
        mounts.append(Mount(source=env["PI_HOST_DATA_DIR"], target="/app/data", type="bind"))
    limits: dict[str, Any] = {}
    if env.get(
        "PI_TASK_MEM_LIMIT"
    ):  # e.g. 6g on the 8 GB VM, so the OOM killer hits the task, not Postgres
        limits["mem_limit"] = env["PI_TASK_MEM_LIMIT"]
    return {
        "image": env.get("PI_PIPELINE_IMAGE", "precursor-pipeline:latest"),
        "docker_url": env.get("PI_DOCKER_URL", "unix://var/run/docker.sock"),
        "network_mode": env.get("PI_DOCKER_NETWORK", "bridge"),  # the network that reaches Postgres
        "environment": environment,
        # hidden in the UI (it holds the metadata DB password)
        "private_environment": {"PI_DATABASE_URL": env.get("PI_DATABASE_URL", "")},
        "mounts": mounts,
        "mount_tmp_dir": False,
        # failed containers are kept for debugging on a laptop; `force` on the VM (they hold PI_DATABASE_URL)
        "auto_remove": env.get("PI_AUTO_REMOVE", "success"),
        "force_pull": False,
        "do_xcom_push": False,  # only tasks that print a result (pending_entries) opt in
        "log_opts_max_size": "10m",
        "log_opts_max_file": "3",
        **limits,
    }
