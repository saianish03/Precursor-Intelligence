"""Release check (node `check_release`; spec §2.0, decision 24): is there work for `cms_nh_pipeline`, and should
this hourly check start it? Read-only, apart from one row appended to `vintage_registry` per recorded check.

Decisions, in precedence order:
  BUSY       a pipeline run holds the `data_pipeline` lock: skip, the next check decides again
  READY      all 7 datasets report one `modified` month, newer than the latest ingested vintage
  RESUME     no new month at CMS, but the last pipeline work did not finish: the latest ingested vintage is not
             published to gold, a pipeline task ran after the last good gold build (the run failed, was killed or
             stopped early), or validation found errors after it. Run again; every node is idempotent
  CHANGED    no new month, but landing has unfinished work: a zip that is new, re-published (another size), not
             downloaded or not yet extracted (e.g. CMS re-posted an old archive, or a corrective partial zip),
             or inventory rows not resolved yet
  PARTIAL    some datasets moved to a newer month, not all 7 to the same one, and nothing else to do: wait; one
             alert once the streak is `release_check.partial_alert_days` old
  NO_CHANGE  nothing to do
  BLOCKED    READY, RESUME or CHANGED that would only repeat the last proceed: that proceed ran the same deployed
             commit less than `release_check.repeat_after_hours` ago, a pipeline run started after it, only BUSY
             or BLOCKED checks came since, and CMS shows nothing that check didn't (no new month, no new or
             re-sized zip). Skip, so a failure that repeats does not start a failing run, and an email, every
             hour. A new commit, new CMS data, the window passing, or a proceed whose run never started (paused
             pipeline, lost event) proceeds; a manual pipeline run always works.

Alerts (the DAG emails `alert`, once): a PARTIAL streak `partial_alert_days` old, and `UNSTARTED_ALERT_AFTER`
proceeds in a row that started no pipeline run (cms_nh_pipeline paused or broken). An alert counts as sent once
decided; a failed send is only logged, as for the anomaly alerts.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from ..common.config import ReleaseCheckCfg
from ..common.io.object_store import ObjectStore
from ..common.io.run_lock import RunLock
from ..gold.build import CURRENT_KEY
from .catalog import default_source, in_scope
from .context import Context
from .inventory import landing_backlog
from .pipeline import LOCK_NAME

log = logging.getLogger(__name__)

PROCEED = ("READY", "RESUME", "CHANGED")
_MODIFIED = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-\d{2}")
# the printed backlog: the JSON line stays a few KB, one short write the DAG reads whole (the row keeps all)
PRINTED_BACKLOG = 20
UNSTARTED_ALERT_AFTER = 3  # proceeds in a row that started no pipeline run: alert once


class ReleaseCheckError(RuntimeError):
    """The metastore answer is incomplete or malformed (exit 1: the DAG retries, then alerts)."""


@dataclass(frozen=True)
class Inputs:
    busy_run: (
        str | None
    )  # the run holding the lock; None when it is free or left stale by a killed run
    modified: dict[str, str]  # dataset id -> YYYY-MM, one per configured dataset
    latest_ingested: str | None  # latest selected Provider Information vintage (resolve's rule)
    latest_published: str | None  # V of gold/_current.json, when gold finished promoting
    backlog: list[str]  # unfinished landing work, "<entry id>@<catalog size>"
    pending_rows: int  # inventory rows not resolved yet
    failure: (
        str | None
    )  # pipeline work after the last good gold build that no good gold build followed
    last_run_start: datetime | None  # when the newest run_log row started (any pipeline task)


@dataclass(frozen=True)
class Decision:
    decision: str
    target: str | None
    reason: str
    alert: str | None = None

    @property
    def proceed(self) -> bool:
        return self.decision in PROCEED

    @property
    def alert_subject(self) -> str | None:
        if not self.alert:
            return None
        what = (
            "CMS release still partial"
            if self.decision == "PARTIAL"
            else "the pipeline did not start"
        )
        return f"[cms_nh_release_check] {what} ({self.decision} {self.target or ''})".replace(
            " )", ")"
        )


def decide(
    inp: Inputs, history: list[dict[str, Any]], *, now: datetime, git_sha: str, cfg: ReleaseCheckCfg
) -> Decision:
    """The decision for ``inp``; ``history`` holds the earlier `vintage_registry` rows, oldest first."""
    if inp.busy_run:
        return Decision("BUSY", None, f"run {inp.busy_run} holds the {LOCK_NAME} lock")
    latest, published = inp.latest_ingested, inp.latest_published
    months = set(inp.modified.values())
    newer = sorted(m for m in months if latest is None or m > latest)
    if len(months) == 1 and newer:
        d = Decision(
            "READY",
            newer[0],
            f"all {len(inp.modified)} datasets at {newer[0]}, newer than the latest ingested ({latest})",
        )
    elif latest is not None and (published is None or published < latest):
        d = Decision(
            "RESUME", latest, f"{latest} is ingested but gold is published for {published}"
        )
    elif latest is not None and inp.failure:
        d = Decision("RESUME", latest, f"the last run did not finish: {inp.failure}")
    elif inp.backlog or inp.pending_rows:
        shown = ", ".join(inp.backlog[:5]) + (" ..." if len(inp.backlog) > 5 else "")
        pending = f"{inp.pending_rows} inventory rows not resolved" if inp.pending_rows else ""
        d = Decision(
            "CHANGED", None, "unfinished landing work: " + "; ".join(filter(None, [shown, pending]))
        )
    elif newer:
        return _partial(inp, newer, history, now, cfg)
    else:
        return Decision(
            "NO_CHANGE", None, f"latest ingested {latest} is published; nothing new at CMS"
        )
    d = _guard(d, inp, history, now, git_sha, cfg)
    return _unstarted(d, inp, history) if d.proceed else d


def _unstarted(d: Decision, inp: Inputs, history: list[dict[str, Any]]) -> Decision:
    """``d`` with an alert, once per streak, when it is the `UNSTARTED_ALERT_AFTER`-th proceed in a row (BUSY and
    BLOCKED checks aside) after which no pipeline run started."""
    streak: list[dict[str, Any]] = []
    for r in reversed(history):
        if r["decision"] in ("BUSY", "BLOCKED"):
            continue
        started = inp.last_run_start and inp.last_run_start > datetime.fromisoformat(
            r["checked_at"]
        )
        if r["decision"] not in PROCEED or started:
            break
        streak.append(r)
    if len(streak) + 1 < UNSTARTED_ALERT_AFTER or any(r.get("alerted") for r in streak):
        return d
    return replace(
        d,
        alert=(
            f"check_release decided to run the pipeline {len(streak) + 1} times in a row since "
            f"{streak[-1]['checked_at']} ({d.decision}: {d.reason}), but no pipeline run started. Is "
            "cms_nh_pipeline paused, or failing to import? This alert is sent once."
        ),
    )


def _guard(
    d: Decision,
    inp: Inputs,
    history: list[dict[str, Any]],
    now: datetime,
    git_sha: str,
    cfg: ReleaseCheckCfg,
) -> Decision:
    """BLOCKED when ``d`` would only repeat the last proceed (module docstring; spec §2.0)."""
    last = next((r for r in reversed(history) if r["decision"] not in ("BUSY", "BLOCKED")), None)
    if last is None or last["decision"] not in PROCEED or (last.get("git_sha") or "") != git_sha:
        return d
    at = datetime.fromisoformat(last["checked_at"])
    if now - at >= timedelta(hours=cfg.repeat_after_hours):
        return d
    if inp.last_run_start is None or inp.last_run_start <= at:
        return d  # its run never started (pipeline paused, VM stopped, event lost): start it now
    new_month = d.decision == "READY" and not (
        last["decision"] == "READY" and last["target"] == d.target
    )
    new_zips = set(inp.backlog) - set(json.loads(last.get("backlog_json") or "[]"))
    if new_month or new_zips:
        return d
    return Decision(
        "BLOCKED",
        d.target,
        f"{d.decision} ({d.reason}) repeats the check at {last['checked_at']}: its run started, on the same "
        f"commit ({git_sha[:12] or 'unknown'}), and CMS shows nothing new since; waiting for a new commit, "
        f"new CMS data or {cfg.repeat_after_hours:g} h",
    )


def _partial(
    inp: Inputs,
    newer: list[str],
    history: list[dict[str, Any]],
    now: datetime,
    cfg: ReleaseCheckCfg,
) -> Decision:
    """PARTIAL, with one alert per streak once it is ``partial_alert_days`` old (BUSY checks do not break it)."""
    streak: list[dict[str, Any]] = []
    for r in reversed(history):
        if r["decision"] == "PARTIAL":
            streak.append(r)
        elif r["decision"] != "BUSY":
            break
    started = datetime.fromisoformat(streak[-1]["checked_at"]) if streak else now
    by_month: dict[str, list[str]] = {}
    for ds, month in sorted(inp.modified.items()):
        by_month.setdefault(month, []).append(ds)
    detail = "; ".join(f"{m}: {', '.join(ds)}" for m, ds in sorted(by_month.items()))
    reason = f"datasets at different months ({detail}); latest ingested {inp.latest_ingested}"
    alert = None
    if not any(r.get("alerted") for r in streak) and now - started >= timedelta(
        days=cfg.partial_alert_days
    ):
        alert = (
            f"The CMS release has been partial since {started.isoformat(timespec='minutes')}: {reason}. "
            "check_release keeps waiting for all 7 datasets; this alert is sent once."
        )
    return Decision("PARTIAL", max(newer), reason, alert)


def latest_published(store: ObjectStore) -> str | None:
    """V of the gold run gold/_current.json names, once it finished promoting. Only the pointer is read: the
    hourly check never hashes gold's files (gold.build.read_current_gold does, before reading gold)."""
    if not store.exists(CURRENT_KEY):
        return None
    with store.open_read(CURRENT_KEY) as f:
        current = json.loads(f.read())
    return current.get("v") if current.get("state") == "current" else None


def last_failure(runs: list[dict[str, Any]], dq_errors: list[dict[str, Any]]) -> str | None:
    """Pipeline work that did not finish: a run_log row after the last good gold build (every run ends with gold
    unless it failed, was killed, or stopped early, and a killed task writes no row of its own, so any later row
    counts), or validation errors after it (dq_results; `validate` writes no run_log row, and validate_gold runs
    after gold). None when nothing ran since, or nothing ran at all."""
    gold_ok = max(
        (r for r in runs if r["task"] == "gold" and r["status"] == "ok"),
        key=lambda r: r["id"],
        default=None,
    )
    if after := [r for r in runs if gold_ok is None or r["id"] > gold_ok["id"]]:
        r = max(after, key=lambda r: r["id"])
        return (
            f"run {r['run_id']} did not reach a good gold build: its last recorded task is {r['task']} "
            f"({r['status']}, {r['finished_at']})"
        )
    since = datetime.fromisoformat(gold_ok["finished_at"]) if gold_ok else None
    errors = [
        r for r in dq_errors if since is None or datetime.fromisoformat(r["checked_at"]) >= since
    ]
    if errors:
        r = max(errors, key=lambda r: r["id"])
        return (
            f"validation found {len(errors)} error(s), the last in run {r['run_id']} "
            f"({r['layer']} {r['table_name']})"
        )
    return None


def gather(ctx: Context, *, now: datetime) -> Inputs:
    """Read CMS (archive catalog and metastore, read-only), then the local state; the lock last, so a run that
    started during the check is BUSY."""
    snapshot = default_source(ctx).fetch()
    expected = {spec.dataset_id for spec in ctx.cfg.tables.values() if spec.dataset_id}
    modified = {
        d["dataset_id"]: str(d["modified"])[:7]
        for d in snapshot.datasets
        if d["dataset_id"] in expected and _MODIFIED.match(str(d.get("modified") or ""))
    }
    if missing := sorted(expected - set(modified)):
        raise ReleaseCheckError(f"metastore gave no valid `modified` for {missing}")

    selected = [
        r for r in ctx.metadata.read("inventory", {"status": "selected"}) if r["vintage_month"]
    ]
    provider = [r["vintage_month"] for r in selected if r["table_name"] == "provider_info"]
    entries = [e for e in snapshot.entries if in_scope(ctx.cfg.scope, e)]
    size = {e.entry_id: e.size_bytes for e in entries}
    lb = landing_backlog(ctx, entries)
    runs = ctx.metadata.read("run_log")
    starts = [datetime.fromisoformat(r["started_at"]) for r in runs if r["started_at"]]
    failure = last_failure(runs, ctx.metadata.read("dq_results", {"severity": "error"}))
    published = latest_published(ctx.object_store)

    holder = RunLock(ctx.metadata, LOCK_NAME).holder()
    stale_after = ctx.settings.env.run_lock.stale_after
    busy = holder["run_id"] if holder else None
    if holder and stale_after and now - datetime.fromisoformat(holder["acquired_at"]) > stale_after:
        log.warning("lock left by run %s since %s is stale: not busy", busy, holder["acquired_at"])
        busy = None
    return Inputs(
        busy_run=busy,
        modified=modified,
        latest_ingested=max(provider or [r["vintage_month"] for r in selected], default=None),
        latest_published=published,
        backlog=sorted({f"{e}@{size[e]}" for e in [*lb.downloads, *lb.extractions]}),
        pending_rows=lb.pending_inventory_rows,
        failure=failure,
        last_run_start=max(starts, default=None),
    )


def run_release_check(
    ctx: Context, *, now: datetime | None = None, record: bool = True
) -> dict[str, Any]:
    """Decide, append the decision to `vintage_registry` (unless ``record`` is False: a manual check by hand
    must not count as the DAG's proceed), and return what the CLI prints as JSON."""
    now = now or datetime.now(UTC)
    inp = gather(ctx, now=now)
    history = sorted(ctx.metadata.read("vintage_registry"), key=lambda r: r["id"])
    git_sha = os.environ.get("PI_GIT_SHA", "")  # the deployed commit, set in the pipeline image
    d = decide(inp, history, now=now, git_sha=git_sha, cfg=ctx.cfg.release_check)
    if record:
        ctx.metadata.append(
            "vintage_registry",
            [
                {
                    "run_id": ctx.run_id,
                    "checked_at": now.isoformat(timespec="seconds"),
                    "decision": d.decision,
                    "target": d.target,
                    "proceed": d.proceed,
                    "alerted": d.alert is not None,
                    "git_sha": git_sha,
                    "latest_ingested": inp.latest_ingested,
                    "latest_published": inp.latest_published,
                    "reason": d.reason,
                    "modified_json": json.dumps(inp.modified, sort_keys=True),
                    "backlog_json": json.dumps(inp.backlog),
                }
            ],
        )
    log.info("check_release: %s (%s)", d.decision, d.reason)
    return {
        "decision": d.decision,
        "proceed": d.proceed,
        "target": d.target,
        "reason": d.reason,
        "alert": d.alert,
        "alert_subject": d.alert_subject,
        "latest_ingested": inp.latest_ingested,
        "latest_published": inp.latest_published,
        "modified": inp.modified,
        "backlog": inp.backlog[:PRINTED_BACKLOG],
        "backlog_count": len(inp.backlog),
        "check_run_id": ctx.run_id,
    }
