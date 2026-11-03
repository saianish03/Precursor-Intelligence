"""precursorintelligence.alerts with a fake SMTP delivery (no network): nothing when no recipient is set, a
failing delivery never raises or logs the password, and the run-failure and anomaly messages."""

import pytest

from precursorintelligence import alerts

RUN = "manual__2026-10-27T10:00:00+00:00"
PASSWORD = "app-password-1234"


@pytest.fixture
def smtp_env(monkeypatch):
    monkeypatch.setenv(alerts.TO_ENV, "oncall@example.com, data@example.com")
    monkeypatch.setenv("PI_ALERT_SMTP_HOST", "smtp.example.com")
    monkeypatch.delenv("PI_ALERT_SMTP_PORT", raising=False)
    monkeypatch.setenv("PI_ALERT_SMTP_USER", "alerts@example.com")
    monkeypatch.setenv("PI_ALERT_SMTP_PASSWORD", PASSWORD)
    monkeypatch.delenv("PI_ALERT_EMAIL_FROM", raising=False)


@pytest.fixture
def sent(monkeypatch, smtp_env):
    """Every message that would have been delivered (no network)."""
    msgs = []
    monkeypatch.setattr(
        alerts, "_deliver", lambda msg, host, port, user, password: msgs.append((msg, host, port))
    )
    return msgs


def test_no_recipient_sends_nothing(monkeypatch):
    monkeypatch.delenv(alerts.TO_ENV, raising=False)
    monkeypatch.setattr(alerts, "_deliver", lambda *a: pytest.fail("sent without a recipient"))
    assert alerts.send("subject", "body") is False


def test_no_smtp_host_sends_nothing(sent, monkeypatch):
    monkeypatch.delenv("PI_ALERT_SMTP_HOST")
    assert alerts.send("subject", "body") is False and sent == []


def test_the_message_goes_to_every_recipient_from_the_smtp_user(sent):
    assert alerts.send("subject", "body") is True
    ((msg, host, port),) = sent
    assert (host, port) == ("smtp.example.com", 587)
    assert msg["To"] == "oncall@example.com, data@example.com"
    assert msg["From"] == "alerts@example.com" and msg["Subject"] == "subject"


def test_a_failing_delivery_never_raises_or_logs_the_password(smtp_env, monkeypatch, caplog):
    def boom(msg, host, port, user, password):
        raise OSError(f"auth failed for {user}:{password}")

    monkeypatch.setattr(alerts, "_deliver", boom)
    assert alerts.send("subject", "body") is False
    assert PASSWORD not in caplog.text


def test_a_failed_run_lists_each_task_with_its_exit_code_and_log():
    failed = [
        ("ingestion.extract_inventory", 2, 1, "http://airflow/x/mapped/2"),
        ("gold.build_gold", None, 137, "http://airflow/gold"),
        ("ingestion.discover_download", None, 3, "http://airflow/discover"),
    ]
    subject, body = alerts.dag_failure_alert("cms_nh_pipeline", RUN, failed, "task_failure")
    assert subject == (
        "[cms_nh_pipeline] run failed: "
        "ingestion.extract_inventory[2], gold.build_gold, ingestion.discover_download"
    )
    assert RUN in body and "Reason: task_failure" in body
    assert (
        "- ingestion.extract_inventory[2]: exit 1 (error)\n  log: http://airflow/x/mapped/2" in body
    )
    assert "- gold.build_gold: exit 137 (killed, usually out of memory)" in body
    assert "- ingestion.discover_download: exit 3 (config or usage error)" in body


def test_a_timed_out_task_has_no_exit_code():
    _, body = alerts.dag_failure_alert(
        "cms_nh_pipeline", RUN, [("gold.build_gold", None, None, "l")], None
    )
    assert "- gold.build_gold: no exit code (timed out" in body and "Reason" not in body


def test_a_run_without_a_failed_task_still_alerts():
    subject, body = alerts.dag_failure_alert("cms_nh_pipeline", RUN, [], "task_failure")
    assert subject == "[cms_nh_pipeline] run failed: see the run"
    assert "No failed task found" in body


def test_anomalies_alert_only_when_there_are_new_ones():
    assert alerts.anomalies_alert("cms_nh_pipeline", RUN, []) is None
    subject, body = alerts.anomalies_alert(
        "cms_nh_pipeline", RUN, ["no_vintage (theme:2026-09-28:nh.zip): NH_X.csv: vintage unknown"]
    )
    assert subject == "[cms_nh_pipeline] 1 new error anomaly"
    assert "- no_vintage (theme:2026-09-28:nh.zip)" in body
    subject, body = alerts.anomalies_alert(
        "cms_nh_pipeline", RUN, [f"no_vintage (e{i}): x" for i in range(25)]
    )
    assert "25 new error anomalies" in subject and "... and 5 more" in body


def test_new_headers_alert_only_when_silver_dropped_columns():
    assert alerts.new_headers_alert("cms_nh_pipeline", RUN, []) is None
    subject, body = alerts.new_headers_alert(
        "cms_nh_pipeline", RUN, ["provider_info 2026-10: High Performing Icon"]
    )
    assert subject == "[cms_nh_pipeline] new CMS columns in 1 silver partition"
    assert "- provider_info 2026-10: High Performing Icon" in body
    assert "column_map.yaml" in body and "ignored:" in body  # what to do about it
    subject, body = alerts.new_headers_alert(
        "cms_nh_pipeline", RUN, [f"provider_info 2026-{i:02d}: X" for i in range(1, 26)]
    )
    assert "25 silver partitions" in subject and "... and 5 more" in body
