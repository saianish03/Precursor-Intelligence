"""The alert callbacks (utils/alerts.py) with a fake SMTP delivery: one email per failed run, naming each failed
task with its exit code and log link, and one for a failed release_lock. The messages and sending themselves
are tested offline in tests/alerts."""

from types import SimpleNamespace

import pytest
from utils import alerts

from precursorintelligence import alerts as messages

RUN = "manual__2026-10-27T10:00:00+00:00"


@pytest.fixture
def smtp_env(monkeypatch):
    monkeypatch.setenv(messages.TO_ENV, "oncall@example.com")
    monkeypatch.setenv("PI_ALERT_SMTP_HOST", "smtp.example.com")


@pytest.fixture
def sent(monkeypatch, smtp_env):
    """Every message that would have been delivered (no network)."""
    msgs = []
    monkeypatch.setattr(
        messages, "_deliver", lambda msg, host, port, user, password: msgs.append((msg, host, port))
    )
    return msgs


class FakeTI:
    def __init__(self, states, exit_codes):
        self.dag_id, self.run_id, self.task_id = "cms_nh_pipeline", RUN, "release_lock"
        self._states, self._codes = states, exit_codes

    def get_task_states(self, dag_id, run_ids):
        return {run_ids[0]: self._states}

    def xcom_pull(self, task_ids, key, map_indexes=None):
        return self._codes.get((task_ids, map_indexes))


def _dag_context(states, exit_codes):
    task_ids = ["ingestion.discover_download", "ingestion.extract_inventory", "gold.build_gold"]
    dag = SimpleNamespace(dag_id="cms_nh_pipeline", task_ids=task_ids)
    return {"dag": dag, "run_id": RUN, "ti": FakeTI(states, exit_codes), "reason": "task_failure"}


def _body(msg):
    return msg.get_content()


def test_a_failed_run_sends_one_alert_with_tasks_exit_codes_and_logs(sent):
    """Run G (bogus entry id): discover_download exits 3, everything after is upstream_failed."""
    states = {
        "ingestion.acquire_lock": "success",
        "ingestion.discover_download": "failed",
        "gold.build_gold": "upstream_failed",
        "release_lock": "success",
    }
    alerts.on_dag_failure(_dag_context(states, {("ingestion.discover_download", None): 3}))
    ((msg, _, _),) = sent
    assert msg["Subject"] == "[cms_nh_pipeline] run failed: ingestion.discover_download"
    body = _body(msg)
    assert RUN in body and "task_failure" in body
    assert "- ingestion.discover_download: exit 3 (config or usage error)" in body
    assert (
        "/dags/cms_nh_pipeline/runs/manual__2026-10-27T10%3A00%3A00%2B00%3A00/tasks/ingestion.discover_download"
        in body
    )
    assert "build_gold" not in body  # upstream_failed is a consequence, not a failure


def test_a_failed_mapped_task_and_an_oom_kill(sent):
    states = {"ingestion.extract_inventory_2": "failed", "gold.build_gold": "failed"}
    codes = {("ingestion.extract_inventory", 2): 1, ("gold.build_gold", None): 137}
    alerts.on_dag_failure(_dag_context(states, codes))
    ((msg, _, _),) = sent
    body = _body(msg)
    assert "- ingestion.extract_inventory[2]: exit 1 (error)" in body
    assert "/tasks/ingestion.extract_inventory/mapped/2" in body
    assert "- gold.build_gold: exit 137 (killed, usually out of memory)" in body


def test_a_timed_out_task_has_no_exit_code(sent):
    alerts.on_dag_failure(_dag_context({"gold.build_gold": "failed"}, {}))
    assert "- gold.build_gold: no exit code (timed out" in _body(sent[0][0])


def test_the_run_alert_is_sent_even_if_the_task_list_is_unavailable(sent):
    context = _dag_context({}, {})
    context["ti"].get_task_states = lambda **kw: (_ for _ in ()).throw(RuntimeError("api down"))
    alerts.on_dag_failure(context)
    assert len(sent) == 1 and "No failed task found" in _body(sent[0][0])


def test_a_failed_release_alerts_although_the_run_is_green(sent):
    alerts.on_release_failure({"ti": FakeTI({}, {}), "exception": RuntimeError("db unreachable")})
    ((msg, _, _),) = sent
    assert "release_lock failed (the run may still show success)" in msg["Subject"]
    body = _body(msg)
    assert "db unreachable" in body and f"lock release --run-id '{RUN}'" in body
