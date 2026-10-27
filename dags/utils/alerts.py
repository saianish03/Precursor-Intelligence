"""Pipeline alerts by email (docs/ALERTS_HANDOFF.md): a failed DAG run, a failed release_lock (which leaves
the run green), and new error anomalies on a run.

SMTP settings come from the environment (docker/airflow-local/.env, never committed): PI_ALERT_EMAIL_TO
(comma-separated), PI_ALERT_SMTP_HOST, PI_ALERT_SMTP_PORT (587 = STARTTLS, 465 = TLS), PI_ALERT_SMTP_USER,
PI_ALERT_SMTP_PASSWORD (for Gmail an app password), PI_ALERT_EMAIL_FROM (default: the user). No recipient
means nothing is sent, so tests and laptop runs stay quiet. The standard library's smtplib, TLS always: no
provider in the image. Callbacks run inside Airflow's own containers (the DAG callback in the dag-processor,
task callbacks in the LocalExecutor worker), which reach the internet. Sending never raises: a lost alert is
logged, and never fails a task or hides the run's own state.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any
from urllib.parse import quote

from utils.pipeline_operator import EXIT_CODE_XCOM, EXIT_KINDS

TO_ENV = "PI_ALERT_EMAIL_TO"
log = logging.getLogger(__name__)


def _deliver(msg: EmailMessage, host: str, port: int, user: str, password: str) -> None:
    tls = ssl.create_default_context()
    if port == 465:
        smtp: smtplib.SMTP = smtplib.SMTP_SSL(host, port, timeout=20, context=tls)
    else:
        smtp = smtplib.SMTP(host, port, timeout=20)
        smtp.starttls(context=tls)  # never send credentials or alerts in clear text
    with smtp:
        if user:
            smtp.login(user, password)
        smtp.send_message(msg)


def send(subject: str, body: str) -> bool:
    """Email the alert; False when no recipient is set or sending failed. Credentials are never logged."""
    env = os.environ
    to = [a.strip() for a in env.get(TO_ENV, "").split(",") if a.strip()]
    if not to:
        log.info("alert not sent (%s unset): %s", TO_ENV, subject)
        return False
    host, user = (
        env.get("PI_ALERT_SMTP_HOST", "").strip(),
        env.get("PI_ALERT_SMTP_USER", "").strip(),
    )
    if not host:
        log.warning("alert not sent: PI_ALERT_SMTP_HOST unset: %s", subject)
        return False
    msg = EmailMessage()
    msg["Subject"], msg["To"] = subject, ", ".join(to)
    msg["From"] = env.get("PI_ALERT_EMAIL_FROM", "").strip() or user
    msg.set_content(body)
    try:
        port = int(env.get("PI_ALERT_SMTP_PORT") or 587)
        _deliver(msg, host, port, user, env.get("PI_ALERT_SMTP_PASSWORD", ""))
    except Exception as exc:  # noqa: BLE001 (an alert must never fail the caller)
        log.warning("alert not sent (%s): %s", exc.__class__.__name__, subject)
        return False
    log.info("alert sent to %d recipient(s): %s", len(to), subject)
    return True


def _base_url() -> str:
    # lazy: keeps this module importable without a configured Airflow
    from airflow.configuration import conf

    return conf.get("api", "base_url", fallback="http://localhost:8080/").rstrip("/")


def log_url(dag_id: str, run_id: str, task_id: str, map_index: int | None = None) -> str:
    mapped = f"/mapped/{map_index}" if map_index is not None and map_index >= 0 else ""
    return f"{_base_url()}/dags/{dag_id}/runs/{quote(run_id, safe='')}/tasks/{task_id}{mapped}"


def exit_text(code: int | None) -> str:
    if code is None:
        return "no exit code (timed out, or failed outside the container)"
    return f"exit {code} ({EXIT_KINDS.get(code, 'unknown')})"


def dag_failure_alert(
    dag_id: str, run_id: str, failed: list[tuple[str, int | None, int | None]], reason: str | None
) -> tuple[str, str]:
    """(subject, body). `failed`: (task_id, map_index, exit code) of each failed task."""
    labels = [t if i is None else f"{t}[{i}]" for t, i, _ in failed]
    subject = f"[{dag_id}] run failed: {', '.join(labels) or 'see the run'}"
    lines = [f"DAG {dag_id}: run {run_id} failed."]
    if reason:
        lines.append(f"Reason: {reason}")
    lines.append("")
    for label, (task_id, map_index, code) in zip(labels, failed, strict=True):
        lines += [
            f"- {label}: {exit_text(code)}",
            f"  log: {log_url(dag_id, run_id, task_id, map_index)}",
        ]
    if not failed:
        lines.append("No failed task found: see the run in the Airflow UI.")
    return subject, "\n".join(lines)


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
    failed: list[tuple[str, int | None, int | None]] = []
    try:
        for task_id, map_index in _failed_tasks(context):
            mapped = {} if map_index is None else {"map_indexes": map_index}
            code = context["ti"].xcom_pull(task_ids=task_id, key=EXIT_CODE_XCOM, **mapped)
            failed.append((task_id, map_index, code if isinstance(code, int) else None))
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


def anomalies_alert(
    dag_id: str, run_id: str, lines: list[str], limit: int = 20
) -> tuple[str, str] | None:
    """(subject, body) for a run's new error anomalies (`precursor-ingest anomalies --new-only`); None if none."""
    if not lines:
        return None
    what = f"{len(lines)} new error {'anomaly' if len(lines) == 1 else 'anomalies'}"
    body = [
        f"DAG {dag_id}: run {run_id} recorded {what}. They do not fail the run; it continues.",
        "",
        *(f"- {line[:300]}" for line in lines[:limit]),
    ]
    if len(lines) > limit:
        body.append(f"... and {len(lines) - limit} more (the anomalies table)")
    return f"[{dag_id}] {what}", "\n".join(body)
