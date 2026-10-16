"""The bronze node: every selected file, in order, with the run-level guards, flags and manifest (spec §3.1)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..common.io.run_lock import RunLock
from ..common.obs.logging import log_context
from ..ingestion.context import Context
from ..ingestion.pipeline import EXIT_OK, EXIT_VALIDATION, LOCK_NAME, task_scope
from .bronze import FAILED, SKIPPED, FileResult, load_file
from .content import PUBLISHED
from .flags import STALE_REFRESH, refresh_stale_refresh

log = logging.getLogger(__name__)

CHECK_NAME = "every_selected_file_has_valid_partition"


@dataclass
class BronzeResult:
    run_id: str
    files: list[FileResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    stale_vintages: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    manifest_key: str = ""

    @property
    def exit_code(self) -> int:
        """Bad data (a failed file, a pending or ambiguous inventory) is exit 2: fix it, don't retry."""
        return EXIT_VALIDATION if self.failures else EXIT_OK


def run_bronze(ctx: Context, *, own_lock: bool = True, now: datetime | None = None) -> BronzeResult:
    """``own_lock=True``: take the run lock for this call (RunLockHeldError if another run holds it).
    ``own_lock=False`` (DAG task): the run must already hold it (RunLockNotHeldError otherwise).
    Only data problems are returned (exit 2); a bug or an outage raises and the CLI turns it into exit 1."""
    now = now or datetime.now(UTC)
    lock = RunLock(ctx.metadata, LOCK_NAME)
    with log_context(run_id=ctx.run_id, env=ctx.settings.env.env):
        if own_lock:
            acquired = lock.acquire(ctx.run_id, stale_after=ctx.settings.env.run_lock.stale_after)
        else:
            lock.require(ctx.run_id)
            acquired = False
        try:
            with task_scope(ctx, "bronze") as detail:
                result = _run(ctx, now)
                detail.update(counts=dict(Counter(f.status for f in result.files)))
                return result
        finally:
            if acquired:
                lock.release(ctx.run_id)


def _run(ctx: Context, now: datetime) -> BronzeResult:
    result = BronzeResult(run_id=ctx.run_id)
    inventory = ctx.metadata.read("inventory")
    pending = sum(1 for r in inventory if r["status"] == "pending")
    selected = sorted(
        (r for r in inventory if r["status"] == "selected"),
        key=lambda r: (r["vintage_month"], r["table_name"]),
    )
    result.failures += _refusals(pending, selected)

    if not result.failures:
        for inv in selected:
            with log_context(table=inv["table_name"], vintage=inv["vintage_month"]):
                outcome = load_file(
                    ctx.object_store, ctx.metadata, ctx.table_sink, inv, run_id=ctx.run_id, now=now
                )
            result.files.append(outcome)
            if outcome.status == FAILED:
                result.failures.append(f"{outcome.table} {outcome.vintage}: {outcome.detail}")
        for vintage in sorted({r["vintage_month"] for r in selected}):
            if refresh_stale_refresh(ctx.metadata, vintage, run_id=ctx.run_id, now=now):
                result.stale_vintages.append(vintage)
        result.missing = files_without_valid_partition(ctx, selected)

    result.manifest_key = _write_manifest(ctx, result, now)
    log.info("bronze finished: %s", dict(Counter(f.status for f in result.files)))
    return result


def _refusals(pending: int, selected: list[dict[str, Any]]) -> list[str]:
    """Reasons to refuse to start, before anything is written."""
    if pending:
        return [f"inventory has {pending} pending rows: finish Step 1 (resolve) first"]
    seen = Counter((r["table_name"], r["vintage_month"]) for r in selected)
    return [
        f"{table} {vintage}: {n} files are selected, expected exactly 1"
        for (table, vintage), n in sorted(seen.items())
        if n > 1
    ]


def files_without_valid_partition(ctx: Context, selected: list[dict[str, Any]]) -> list[str]:
    """The over-every-file check, not only the files touched in this run: each selected file needs a
    loaded or reloaded row for its current zip_sha256 and a partition that still exists."""
    done = {
        (r["table_name"], r["vintage_month"]): r
        for r in sorted(
            ctx.metadata.read("bronze_files", {"status": list(PUBLISHED)}), key=lambda r: r["id"]
        )
    }
    partitions = {t: ctx.table_sink.list_partitions(t) for t in {r["table_name"] for r in selected}}
    missing = []
    for inv in selected:
        row = done.get((inv["table_name"], inv["vintage_month"]))
        has_partition = {"vintage": inv["vintage_month"]} in partitions[inv["table_name"]]
        if row is None or row["zip_sha256"] != inv["zip_sha256"] or not has_partition:
            missing.append(f"{inv['table_name']} {inv['vintage_month']}")
    return missing


def _write_manifest(ctx: Context, result: BronzeResult, now: datetime) -> str:
    """One JSON per run under metadata/bronze/. The timestamp in the name means a retry never overwrites it."""
    stale = {
        r["vintage_month"]
        for r in ctx.metadata.read("vintage_flags", {"flag": STALE_REFRESH})
        if r["value"]
    }
    manifest = {
        "run_id": ctx.run_id,
        "started_at": ctx.started_at,
        "finished_at": now.isoformat(timespec="seconds"),
        "exit_code": result.exit_code,
        "counts": dict(Counter(f.status for f in result.files)),
        "files": [
            {"table": f.table, "vintage": f.vintage, "status": f.status, "detail": f.detail}
            for f in result.files
            if f.status != SKIPPED
        ],
        "revised": [{"table": f.table, "vintage": f.vintage} for f in result.files if f.revised],
        "stale_refresh_vintages": sorted(stale),
        CHECK_NAME: {"passed": not result.missing, "missing": result.missing},
        "failures": result.failures,
    }
    stamp = datetime.now(
        UTC
    )  # the real clock, to the microsecond: a retry in the same second cannot collide
    key = f"metadata/bronze/run_date={stamp:%Y-%m-%d}/bronze_run_{ctx.run_id}_{stamp:%Y%m%dT%H%M%S%fZ}.json"
    ctx.object_store.put_bytes(key, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    return key
