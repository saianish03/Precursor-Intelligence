"""PipelineDockerOperator on the pinned Docker provider: exit 0 returns the last output line (the XCom), exit 1
stays a retryable DockerContainerFailedException, exits 2 and 3 become AirflowFailException (no retry).
The Docker API is mocked at the provider's `cli` (docker.APIClient), so this runs without a daemon."""

from importlib.metadata import version
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from airflow.exceptions import AirflowFailException
from airflow.providers.docker.exceptions import DockerContainerFailedException
from utils.pipeline_operator import (
    PipelineDockerOperator,
    docker_kwargs,
    exit_code_of,
    last_json_list,
)


def _run(status_code: int, logs: list[bytes]):
    cli = MagicMock()
    cli.images.return_value = [{"Id": "img"}]
    cli.create_container.return_value = {"Id": "c1"}
    cli.attach.return_value = iter(logs)
    cli.wait.return_value = {"Error": None, "StatusCode": status_code}
    op = PipelineDockerOperator(
        task_id="t",
        image="precursor-pipeline:test",
        command="precursor-ingest entries --pending",
        do_xcom_push=True,
        mount_tmp_dir=False,
    )
    with patch.object(PipelineDockerOperator, "cli", new_callable=PropertyMock, return_value=cli):
        return op.execute(context={})


def test_pinned_provider_version():
    assert (
        version("apache-airflow-providers-docker") == "4.6.1"
    )  # the message format below was checked on it


def test_exit_0_returns_the_last_line():
    assert _run(0, [b"log line\n", b'["theme:2021-07-27:nursing-homes_2021-07-27.zip"]\n']) == (
        '["theme:2021-07-27:nursing-homes_2021-07-27.zip"]'
    )


def test_exit_1_is_retryable():
    with pytest.raises(DockerContainerFailedException) as exc:
        _run(1, [b"boom\n"])
    assert not isinstance(exc.value, AirflowFailException)
    assert exit_code_of(exc.value) == 1


@pytest.mark.parametrize("code", [2, 3])
def test_exit_2_and_3_fail_without_retry(code):
    with pytest.raises(AirflowFailException, match=f"pipeline exit {code}"):
        _run(code, [b"validation failed\n"])


def test_docker_kwargs_from_environment(monkeypatch):
    monkeypatch.setenv("PI_PIPELINE_IMAGE", "precursor-pipeline:x")
    monkeypatch.setenv("PI_HOST_DATA_DIR", "/srv/pi/data")
    monkeypatch.delenv("PI_BUCKET_URI", raising=False)
    kw = docker_kwargs()
    assert kw["image"] == "precursor-pipeline:x"
    assert [(m["Source"], m["Target"]) for m in kw["mounts"]] == [("/srv/pi/data", "/app/data")]
    assert "PI_BUCKET_URI" not in kw["environment"]  # unset, so prod.yaml's default bucket applies

    assert kw["do_xcom_push"] is False and "mem_limit" not in kw
    assert (kw["log_opts_max_size"], kw["log_opts_max_file"]) == ("10m", "3")

    monkeypatch.setenv("PI_TASK_MEM_LIMIT", "6g")
    assert docker_kwargs()["mem_limit"] == "6g"
    monkeypatch.delenv("PI_HOST_DATA_DIR")
    monkeypatch.setenv("PI_BUCKET_URI", "gs://bucket")
    kw = docker_kwargs()
    assert kw["mounts"] == [] and kw["environment"]["PI_BUCKET_URI"] == "gs://bucket"


def test_the_result_list_survives_log_lines_after_it():
    """stdout and stderr arrive interleaved, so a JSON log line can follow the result (prod logs are JSON)."""
    lines = [
        '{"ts": "…", "level": "INFO", "msg": "entries: 2 pending"}',
        '["theme:2021-07-27:nursing-homes_2021-07-27.zip", "current:2026-10-08:theme_nursing-homes_current.zip"]',
        '{"ts": "…", "level": "INFO", "msg": "task entries ok in 0.4s"}',
    ]
    assert last_json_list(lines) == [
        "theme:2021-07-27:nursing-homes_2021-07-27.zip",
        "current:2026-10-08:theme_nursing-homes_current.zip",
    ]
    assert last_json_list("\n".join(lines)) == last_json_list(lines)
    assert (
        last_json_list(["[]"]) == []
        and last_json_list(None) == []
        and last_json_list(["no json"]) == []
    )
