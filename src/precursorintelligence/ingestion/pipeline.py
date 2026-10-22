"""Step 1 pipeline runner.

Each task is a plain function of the Context. ``run_step1`` chains them; a DAG orchestrator (Airflow)
can instead call the same functions as separate tasks, mapping download/extract/inventory over the
selected entries. Every task invocation is wrapped by ``task_scope`` which sets the log context,
times it and writes start/finish rows to ``run_log``. Run lock (D8): a standalone run takes and releases
it itself; a DAG task (``own_lock=False``) requires that its run already holds it (``lock acquire``).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..common.io.run_lock import RunLock
from ..common.obs.logging import log_context
from .catalog import discover, select
from .context import Context, utc_now
from .download import run_download
from .extract import extract_entry
from .http import UnsafeUrlError
from .inventory import inventory_entry, iter_good_downloads
from .report import report
from .resolve import resolve
from .sources.base import CatalogContractError
from .zipsafe import UnsafeArchiveError

log = logging.getLogger(__name__)

STEPS = ("discover", "download", "extract", "inventory", "resolve", "report")
LOCK_NAME = "data_pipeline"  # shared by every data step (Step 2 takes the same lock)

# Exit codes (CLI, Airflow): 0 ok or nothing new; 1 error (bug, network, storage; retrying may help);
# 2 validation / drift (bad or changed data: fix the data or the config, don't just retry); 3 config/usage.
EXIT_OK, EXIT_ERROR, EXIT_VALIDATION, EXIT_CONFIG = 0, 1, 2, 3
VALIDATION_DOWNLOAD_STATUSES = {"quarantined", "rejected"}
VALIDATION_ERRORS = (UnsafeArchiveError, UnsafeUrlError, CatalogContractError)


@contextmanager
def task_scope(
    ctx: Context, task: str, entry_id: str | None = None, **ctx_fields: Any
) -> Iterator[dict[str, Any]]:
    detail: dict[str, Any] = {}
    started, t0 = utc_now(), time.monotonic()
    with log_context(task=task, entry_id=entry_id, **ctx_fields):
        log.debug("task %s started", task)
        status = "ok"
        try:
            yield detail
        except Exception as exc:
            status, detail["error"] = "failed", f"{exc.__class__.__name__}: {exc}"
            log.exception("task %s failed", task)
            raise
        finally:
            dur = time.monotonic() - t0
            ctx.metadata.append(
                "run_log",
                [
                    {
                        "run_id": ctx.run_id,
                        "task": task,
                        "entry_id": entry_id,
                        "status": status,
                        "started_at": started,
                        "finished_at": utc_now(),
                        "duration_s": f"{dur:.2f}",
                        "detail": json.dumps(detail, default=str),
                    }
                ],
            )
            if entry_id is None:
                log.info(
                    "task %s %s in %.1fs %s", task, status, dur, json.dumps(detail, default=str)
                )


@dataclass
class Step1Result:
    run_id: str
    selected: int = 0
    downloads: dict[str, int] = field(default_factory=dict)
    extracted: int = 0
    inventoried: int = 0
    reports: dict[str, str] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    n_validation_failures: int = 0  # how many of ``failures`` are data problems, not errors

    def fail(self, message: str, *, validation: bool) -> None:
        self.failures.append(message)
        self.n_validation_failures += validation

    @property
    def exit_code(self) -> int:
        """Any error wins (a retry may fix it); otherwise 2 if only data problems were found."""
        if len(self.failures) > self.n_validation_failures:
            return EXIT_ERROR
        return EXIT_VALIDATION if self.failures else EXIT_OK


def run_step1(
    ctx: Context,
    *,
    steps: tuple[str, ...] = STEPS,
    zip_dates: list[date] | None = None,
    limit: int | None = None,
    entry_ids: list[str] | None = None,
    own_lock: bool = True,
) -> Step1Result:
    """``own_lock=True``: take the run lock for this call (RunLockHeldError if another run holds it).
    ``own_lock=False`` (DAG tasks): the run must already hold it (RunLockNotHeldError otherwise)."""
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; valid: {STEPS}")
    lock = RunLock(ctx.metadata, LOCK_NAME)
    with log_context(run_id=ctx.run_id, env=ctx.settings.env.env):
        if own_lock:
            acquired = lock.acquire(ctx.run_id, stale_after=ctx.settings.env.run_lock.stale_after)
        else:
            lock.require(ctx.run_id)
            acquired = False
        try:
            return _run_steps(ctx, steps, zip_dates, limit, entry_ids)
        finally:
            if acquired:
                lock.release(ctx.run_id)


def _run_steps(
    ctx: Context,
    steps: tuple[str, ...],
    zip_dates: list[date] | None,
    limit: int | None,
    entry_ids: list[str] | None,
) -> Step1Result:
    res = Step1Result(run_id=ctx.run_id)
    log.info(
        "Step 1 run %s starting: steps=%s, store=%s, metadata=%s",
        ctx.run_id,
        ",".join(steps),
        ctx.object_store.root_uri,
        getattr(ctx.metadata, "safe_url", "?"),
    )
    if "discover" in steps:
        with task_scope(ctx, "discover") as d:
            out = discover(ctx)
            d.update(
                entries=out.n_entries,
                new=out.n_new,
                removed=out.n_removed,
                size_changed=out.n_size_changed,
            )

    with task_scope(ctx, "select") as d:
        entries = select(ctx, zip_dates=zip_dates, limit=limit, entry_ids=entry_ids)
        d.update(selected=len(entries))
        res.selected = len(entries)

    if "download" in steps and entries:
        with task_scope(ctx, "download") as d:
            results = run_download(ctx, entries)
            for r in results:
                res.downloads[r.status] = res.downloads.get(r.status, 0) + 1
                if not r.ok:
                    res.fail(
                        f"download {r.entry.entry_id}: {r.status} {r.detail}",
                        validation=r.status in VALIDATION_DOWNLOAD_STATUSES,
                    )
            d.update(res.downloads)

    good = list(iter_good_downloads(ctx, entries)) if entries else []
    extract_failed: set[str] = set()
    if "extract" in steps:
        for entry, row in good:
            try:
                with task_scope(ctx, "extract", entry.entry_id, zip_name=entry.zip_name) as d:
                    xr = extract_entry(ctx, entry, row["object_key"], row["sha256"])
                    d.update(written=xr.n_extracted, existing=xr.n_skipped_existing)
                    res.extracted += xr.n_extracted
            except Exception as exc:
                extract_failed.add(entry.entry_id)
                res.fail(
                    f"extract {entry.entry_id}: {exc}",
                    validation=isinstance(exc, VALIDATION_ERRORS),
                )

    if "inventory" in steps:
        for entry, row in good:
            if entry.entry_id in extract_failed:
                continue  # an inventory without the extracted files would look complete
            try:
                with task_scope(ctx, "inventory", entry.entry_id, zip_name=entry.zip_name) as d:
                    rows = inventory_entry(ctx, entry, row["object_key"], row["sha256"])
                    d.update(members=len(rows))
                    res.inventoried += 1
            except Exception as exc:
                res.fail(
                    f"inventory {entry.entry_id}: {exc}",
                    validation=isinstance(exc, VALIDATION_ERRORS),
                )

    resolved = None
    if "resolve" in steps:
        with task_scope(ctx, "resolve") as d:
            resolved = resolve(ctx)
            d.update(
                latest_vintage=resolved.latest_vintage,
                duplicates=len(resolved.duplicates),
                anomalies=len(resolved.anomalies),
            )
    if "report" in steps:
        with task_scope(ctx, "report") as d:
            resolved = resolved or resolve(ctx)
            res.reports = report(ctx, resolved)
            d.update(reports=list(res.reports.values()))
    for a in resolved.anomalies if resolved else []:
        if a["kind"] == "no_tables_recognised":
            res.fail(a["detail"], validation=True)  # a data problem: fix the patterns, don't retry

    if res.failures:
        log.warning("Step 1 run %s finished with %d failures", ctx.run_id, len(res.failures))
    else:
        log.info("Step 1 run %s finished OK", ctx.run_id)
    return res
