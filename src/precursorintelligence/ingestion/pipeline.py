"""Step 1 pipeline runner.

Each task is a plain function of the Context. ``run_step1`` chains them; a DAG orchestrator (Prefect /
Airflow) can instead call the same functions as separate tasks, mapping download/extract/inventory over
the selected entries. Every task invocation is wrapped by ``task_scope`` which sets the log context,
times it and writes start/finish rows to ``run_log``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date

from ..common.context import Context, utc_now
from .catalog import discover, select
from .download import run_download
from .extract import extract_entry
from .inventory import inventory_entry, iter_good_downloads
from .report import report
from .resolve import resolve
from ..common.obs.logging import log_context

log = logging.getLogger(__name__)

STEPS = ("discover", "download", "extract", "inventory", "resolve", "report")


@contextmanager
def task_scope(ctx: Context, task: str, entry_id: str | None = None, **ctx_fields) -> Iterator[dict]:
    detail: dict = {}
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
            ctx.metadata.append("run_log", [{
                "run_id": ctx.run_id, "task": task, "entry_id": entry_id, "status": status, "started_at": started,
                "finished_at": utc_now(), "duration_s": f"{dur:.2f}", "detail": json.dumps(detail, default=str),
            }])
            if entry_id is None:
                log.info("task %s %s in %.1fs %s", task, status, dur, json.dumps(detail, default=str))


@dataclass
class Step1Result:
    run_id: str
    selected: int = 0
    downloads: dict[str, int] = field(default_factory=dict)
    extracted: int = 0
    inventoried: int = 0
    reports: dict[str, str] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def run_step1(ctx: Context, *, steps: tuple[str, ...] = STEPS, zip_dates: list[date] | None = None,
              limit: int | None = None) -> Step1Result:
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; valid: {STEPS}")
    res = Step1Result(run_id=ctx.run_id)
    with log_context(run_id=ctx.run_id, env=ctx.settings.env.env):
        log.info("Step 1 run %s starting: steps=%s, store=%s, metadata=%s", ctx.run_id, ",".join(steps),
                 ctx.object_store.root_uri, getattr(ctx.metadata, "safe_url", "?"))
        if "discover" in steps:
            with task_scope(ctx, "discover") as d:
                out = discover(ctx)
                d.update(entries=out.n_entries, new=out.n_new, removed=out.n_removed, size_changed=out.n_size_changed)

        with task_scope(ctx, "select") as d:
            entries = select(ctx, zip_dates=zip_dates, limit=limit)
            d.update(selected=len(entries))
            res.selected = len(entries)

        if "download" in steps and entries:
            with task_scope(ctx, "download") as d:
                results = run_download(ctx, entries)
                for r in results:
                    res.downloads[r.status] = res.downloads.get(r.status, 0) + 1
                    if not r.ok:
                        res.failures.append(f"download {r.entry.entry_id}: {r.status} {r.detail}")
                d.update(res.downloads)

        good = list(iter_good_downloads(ctx, entries)) if entries else []
        if "extract" in steps:
            for entry, row in good:
                try:
                    with task_scope(ctx, "extract", entry.entry_id, zip_name=entry.zip_name) as d:
                        r = extract_entry(ctx, entry, row["object_key"], row["sha256"])
                        d.update(written=r.n_extracted, existing=r.n_skipped_existing)
                        res.extracted += r.n_extracted
                except Exception as exc:
                    res.failures.append(f"extract {entry.entry_id}: {exc}")

        if "inventory" in steps:
            for entry, row in good:
                try:
                    with task_scope(ctx, "inventory", entry.entry_id, zip_name=entry.zip_name) as d:
                        rows = inventory_entry(ctx, entry, row["object_key"], row["sha256"])
                        d.update(members=len(rows))
                        res.inventoried += 1
                except Exception as exc:
                    res.failures.append(f"inventory {entry.entry_id}: {exc}")

        resolved = None
        if "resolve" in steps:
            with task_scope(ctx, "resolve") as d:
                resolved = resolve(ctx)
                d.update(latest_vintage=resolved.latest_vintage, duplicates=len(resolved.duplicates),
                         anomalies=len(resolved.anomalies))
        if "report" in steps:
            with task_scope(ctx, "report") as d:
                resolved = resolved or resolve(ctx)
                res.reports = report(ctx, resolved)
                d.update(reports=list(res.reports.values()))

        if res.failures:
            log.warning("Step 1 run %s finished with %d failures", ctx.run_id, len(res.failures))
        else:
            log.info("Step 1 run %s finished OK", ctx.run_id)
    return res
