"""The silver node, Steps 2b-2e: every bronze partition, citation lookup versions, the every-partition check,
the Checkpoint 2 report and the run manifest (docs/DATA_PIPELINE_SPEC.md §3.2-3.5).

Ledgers (Step 3), the facility universe and QM inventory (Step 5) and the calibration tasks come after
Checkpoint 2 is approved.
"""

from __future__ import annotations

import io
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

import polars as pl

from ..common.io.run_lock import RunLock
from ..common.io.table_sink import ParquetTableSink, TableSink
from ..common.obs.logging import log_context
from ..ingestion.context import Context
from ..ingestion.pipeline import EXIT_OK, EXIT_VALIDATION, LOCK_NAME, task_scope
from .columns import SilverConfig, load_silver_config
from .ledgers import published_vintages
from .pipeline import files_without_valid_partition
from .silver import FAILED, SKIPPED, PartitionResult, build_partition, is_current, latest_bronze
from .silver_report import write_reports
from .silver_steps import build_ledgers_and_universe, run_calibration

log = logging.getLogger(__name__)

CHECK_NAME = "every_bronze_partition_has_current_silver"
LOOKUP_VERSIONS_KEY = "silver/citation_lookup_versions/part-0.parquet"
LOOKUP_CONTENT = ("deficiency_prefix_and_number", "deficiency_description", "deficiency_category")


@dataclass
class SilverResult:
    run_id: str
    partitions: list[PartitionResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    lookup_versions: int = 0
    tables: dict[str, int] = field(
        default_factory=dict
    )  # all-vintage tables (Steps 3-5): rows written
    manifest_key: str = ""
    report_keys: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return EXIT_VALIDATION if self.failures else EXIT_OK


def silver_sink(ctx: Context) -> TableSink:
    return ParquetTableSink(ctx.object_store, prefix="silver")


def run_silver(
    ctx: Context,
    *,
    own_lock: bool = True,
    cfg: SilverConfig | None = None,
    calibrate: bool = False,
    now: datetime | None = None,
) -> SilverResult:
    """Same lock contract as bronze: ``own_lock=False`` (DAG task) requires the run to hold it already.
    A broken config raises ConfigError before the lock is taken."""
    cfg = cfg or load_silver_config()
    now = now or datetime.now(UTC)
    lock = RunLock(ctx.metadata, LOCK_NAME)
    with log_context(run_id=ctx.run_id, env=ctx.settings.env.env):
        if own_lock:
            acquired = lock.acquire(ctx.run_id, stale_after=ctx.settings.env.run_lock.stale_after)
        else:
            lock.require(ctx.run_id)
            acquired = False
        try:
            with task_scope(ctx, "silver") as detail:
                result = _run(ctx, cfg, now, calibrate)
                detail.update(counts=dict(Counter(p.status for p in result.partitions)))
                return result
        finally:
            if acquired:
                lock.release(ctx.run_id)


def _run(ctx: Context, cfg: SilverConfig, now: datetime, calibrate: bool) -> SilverResult:
    result = SilverResult(run_id=ctx.run_id)
    inventory = ctx.metadata.read("inventory")
    pending = sum(1 for r in inventory if r["status"] == "pending")
    selected = [r for r in inventory if r["status"] == "selected"]
    if pending:
        result.failures.append(f"inventory has {pending} pending rows: finish Step 1 first")
    elif missing := files_without_valid_partition(ctx, selected):
        result.failures.append(
            f"bronze is incomplete ({len(missing)} files), e.g. {missing[:3]}: run bronze first"
        )

    sink = silver_sink(ctx)
    if not result.failures:
        for bronze in latest_bronze(ctx.metadata):
            if bronze["table_name"] not in cfg.sources:
                continue
            with log_context(table=bronze["table_name"], vintage=bronze["vintage_month"]):
                outcome = build_partition(
                    ctx.table_sink, sink, ctx.metadata, bronze, cfg, run_id=ctx.run_id, now=now
                )
            result.partitions.append(outcome)
            if outcome.status == FAILED:
                result.failures.append(f"{outcome.source} {outcome.vintage}: {outcome.detail}")
        result.lookup_versions = write_lookup_versions(ctx, sink)
        result.missing = partitions_without_current_silver(ctx, sink, cfg)
        if result.missing and not any(p.status == FAILED for p in result.partitions):
            result.failures.append(f"{CHECK_NAME} failed for {len(result.missing)} partitions")
        if not result.failures:
            # Steps 3-5 read every vintage, so they run only on complete, current stacked tables.
            vintages = published_vintages(sink)
            with task_scope(ctx, "silver_ledgers"):
                built = build_ledgers_and_universe(ctx, vintages)
            result.tables = built["counts"]
            if calibrate:
                with task_scope(ctx, "silver_calibrate"):
                    result.report_keys += run_calibration(ctx, vintages, built)

    result.report_keys = (
        write_reports(ctx, cfg, result.partitions, result.lookup_versions) + result.report_keys
    )
    result.manifest_key = _write_manifest(ctx, cfg, result, now)
    log.info("silver finished: %s", dict(Counter(p.status for p in result.partitions)))
    return result


def lookup_versions(lookup: pl.DataFrame, published: list[str]) -> pl.DataFrame:
    """One row per contiguous run of published vintages in which a (prefix, tag) has identical content
    (spec §3.5): a change A -> B -> A gives three versions, and so does a gap in the tag's presence."""
    index = {v: i for i, v in enumerate(published)}
    rows = (
        lookup.with_columns(
            pl.col("vintage_month").replace_strict(index, return_dtype=pl.Int64).alias("_i")
        )
        .sort(["deficiency_prefix", "deficiency_tag", "_i"])
        .with_columns(
            pl.concat_str(
                [pl.col(c).fill_null("") for c in LOOKUP_CONTENT], separator="\x1f"
            ).alias("_c")
        )
    )
    key = ["deficiency_prefix", "deficiency_tag"]
    rows = rows.with_columns(
        (
            (pl.col("_i").diff().over(key).fill_null(0) != 1)
            | (pl.col("_c") != pl.col("_c").shift(1).over(key)).fill_null(True)
        )
        .cum_sum()
        .over(key)
        .alias("_v")
    )
    return (
        rows.group_by([*key, "_v"], maintain_order=True)
        .agg(
            *[pl.col(c).first() for c in LOOKUP_CONTENT],
            pl.col("vintage_month").first().alias("valid_from_vintage"),
            pl.col("vintage_month").last().alias("valid_to_vintage"),
        )
        .drop("_v")
        .sort([*key, "valid_from_vintage"])
    )


def write_lookup_versions(ctx: Context, sink: TableSink) -> int:
    lookup = sink.read("citation_lookup")
    if lookup.is_empty():
        return 0
    published = sorted(lookup["vintage_month"].unique().to_list())
    versions = lookup_versions(lookup.drop("vintage"), published)
    buf = io.BytesIO()
    versions.write_parquet(buf, compression="zstd")
    ctx.object_store.put_bytes(LOOKUP_VERSIONS_KEY, buf.getvalue(), overwrite=True)
    return versions.height


def partitions_without_current_silver(
    ctx: Context, sink: TableSink, cfg: SilverConfig
) -> list[str]:
    """Over every bronze partition, not only those touched in this run: each silver output exists and was
    built from the current bronze content with the current configs."""
    return [
        f"{t.name} {b['vintage_month']}"
        for b in latest_bronze(ctx.metadata)
        if b["table_name"] in cfg.sources
        for t in cfg.tables_for(b["table_name"])
        if not is_current(ctx.metadata, sink, t.name, b, cfg.config_hash)
    ]


def _write_manifest(ctx: Context, cfg: SilverConfig, result: SilverResult, now: datetime) -> str:
    manifest = {
        "run_id": ctx.run_id,
        "started_at": ctx.started_at,
        "finished_at": now.isoformat(timespec="seconds"),
        "exit_code": result.exit_code,
        "config_hash": cfg.config_hash,
        "counts": dict(Counter(p.status for p in result.partitions)),
        "partitions": [
            {"table": p.source, "vintage": p.vintage, "status": p.status, "detail": p.detail}
            for p in result.partitions
            if p.status != SKIPPED
        ],
        "citation_lookup_versions": result.lookup_versions,
        "all_vintage_tables": result.tables,
        CHECK_NAME: {"passed": not result.missing, "missing": result.missing},
        "reports": result.report_keys,
        "failures": result.failures,
    }
    stamp = datetime.now(UTC)  # the real clock: a retry in the same second cannot collide
    key = f"metadata/silver/run_date={stamp:%Y-%m-%d}/silver_run_{ctx.run_id}_{stamp:%Y%m%dT%H%M%S%fZ}.json"
    ctx.object_store.put_bytes(key, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    return key
