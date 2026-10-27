"""Pipeline alerts by email (docs/ALERTS_HANDOFF.md): a failed DAG run, a failed release_lock (which leaves
the run green), and new error anomalies on a run. The Airflow callbacks that call this live in
dags/utils/alerts.py.

SMTP settings come from the environment (docker/airflow-local/.env, never committed): PI_ALERT_EMAIL_TO
(comma-separated), PI_ALERT_SMTP_HOST, PI_ALERT_SMTP_PORT (587 = STARTTLS, 465 = TLS), PI_ALERT_SMTP_USER,
PI_ALERT_SMTP_PASSWORD (for Gmail an app password), PI_ALERT_EMAIL_FROM (default: the user). No recipient
means nothing is sent, so tests and laptop runs stay quiet. The standard library only (smtplib, TLS always),
so the Airflow image needs no extra dependency: it mounts this package and imports only this module. Sending
never raises: a lost alert is logged, and never fails a task or hides the run's own state.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

TO_ENV = "PI_ALERT_EMAIL_TO"
# A pipeline container's exit codes (ingestion/pipeline.py, plus 137 from Docker), as alerts name them.
EXIT_KINDS = {
    1: "error",
    2: "data or validation failure",
    3: "config or usage error",
    137: "killed, usually out of memory",
}
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


def exit_text(code: int | None) -> str:
    if code is None:
        return "no exit code (timed out, or failed outside the container)"
    return f"exit {code} ({EXIT_KINDS.get(code, 'unknown')})"


def dag_failure_alert(
    dag_id: str,
    run_id: str,
    failed: list[tuple[str, int | None, int | None, str]],
    reason: str | None,
) -> tuple[str, str]:
    """(subject, body). `failed`: (task_id, map_index, exit code, log link) of each failed task."""
    labels = [t if i is None else f"{t}[{i}]" for t, i, _, _ in failed]
    subject = f"[{dag_id}] run failed: {', '.join(labels) or 'see the run'}"
    lines = [f"DAG {dag_id}: run {run_id} failed."]
    if reason:
        lines.append(f"Reason: {reason}")
    lines.append("")
    for label, (_, _, code, link) in zip(labels, failed, strict=True):
        lines += [f"- {label}: {exit_text(code)}", f"  log: {link}"]
    if not failed:
        lines.append("No failed task found: see the run in the Airflow UI.")
    return subject, "\n".join(lines)


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
