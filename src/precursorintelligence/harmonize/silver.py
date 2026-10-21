"""Build the silver partitions of one bronze (table, vintage) partition (docs/DATA_PIPELINE_SPEC.md §3.2-3.5).

One bronze partition feeds one silver table, except provider_info, which feeds provider_by_vintage and
provider_identity_by_vintage. Either every output of a partition is written, or none is (fail closed).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import polars as pl

from ..common.io.metadata_store import MetadataStore
from ..common.io.table_sink import TableSink
from .columns import Resolution, SilverConfig, SilverTable, resolve_headers
from .content import PUBLISHED
from .normalize import (
    NormalizeError,
    measure_period_bounds,
    normalize,
    null_column,
    ownership_group,
)

log = logging.getLogger(__name__)

BUILT, FAILED, SKIPPED = "built", "failed", "skipped"
PENALTY_FIELDS = ("ccn", "penalty_date", "penalty_type", "fine_amount", "payment_denial_start_date")
# Event date column per table, for the post-cutoff count (spec §3.5 "Events after the cutoff").
EVENT_DATES = {
    "health_citations": "survey_date",
    "survey_dates": "survey_date",
    "penalties": "penalty_date",
}


class PartitionError(ValueError):
    """Bad data in one partition: nothing is written for it and the node exits 2."""


@dataclass
class PartitionResult:
    source: str
    vintage: str
    status: str  # built | failed | skipped
    detail: str = ""
    problems: list[tuple[str, str]] = field(default_factory=list)  # header problems, for the report
    stats: dict[str, Any] = field(default_factory=dict)


def latest_bronze(meta: MetadataStore) -> list[dict[str, Any]]:
    """The current bronze row per (table, vintage): the latest loaded or reloaded one."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for r in meta.read("bronze_files", {"status": list(PUBLISHED)}):
        k = (r["table_name"], r["vintage_month"])
        if k not in latest or r["id"] > latest[k]["id"]:
            latest[k] = r
    return sorted(latest.values(), key=lambda r: (r["vintage_month"], r["table_name"]))


def _latest_built(meta: MetadataStore, table: str, vintage: str) -> dict[str, Any] | None:
    rows = meta.read(
        "silver_partitions", {"table_name": table, "vintage_month": vintage, "status": BUILT}
    )
    return max(rows, key=lambda r: r["id"]) if rows else None


def is_current(
    meta: MetadataStore, sink: TableSink, table: str, bronze: Mapping[str, Any], config_hash: str
) -> bool:
    """Spec §3.5 skip key: the latest built row has the same bronze content hash, zip and config hash,
    and the partition still exists."""
    row = _latest_built(meta, table, bronze["vintage_month"])
    return (
        row is not None
        and row["content_hash"] == bronze["content_hash"]
        and row["zip_sha256"] == bronze["zip_sha256"]
        and row["config_hash"] == config_hash
        and {"vintage": bronze["vintage_month"]} in sink.list_partitions(table)
    )


def harmonize(
    frame: pl.DataFrame, source: str, vintage: str, cfg: SilverConfig
) -> tuple[dict[str, pl.DataFrame], dict[str, Any], Resolution]:
    """Map, type, derive and select the silver tables of one bronze partition. Raises PartitionError."""
    smap = cfg.sources[source]
    data_cols = [c for c in frame.columns if not c.startswith("_") and c != "vintage"]
    res = resolve_headers(smap, data_cols, vintage)
    if res.problems:
        raise PartitionError("; ".join(f"{h}: {p}" for h, p in res.problems))

    renamed = frame.select(
        [pl.col(raw).alias(canon) for canon, raw in res.rename.items()] + [pl.col("_row_number")]
    ).sort("_row_number")
    dtypes = {c: smap.columns[c].dtype for c in res.rename}
    try:
        typed, invalid = normalize(renamed, dtypes, cfg.value_maps)
    except NormalizeError as exc:
        raise PartitionError(str(exc)) from None
    typed = typed.with_columns(
        [null_column(c, smap.columns[c].dtype, typed.height) for c in res.absent]
    )

    cutoff = date.fromisoformat(f"{vintage}-01")
    if "processing_date" in smap.columns:
        wrong = (
            typed.filter(pl.col("processing_date") != cutoff).height
            + typed["processing_date"].null_count()
        )
        if wrong:
            raise PartitionError(f"{wrong} rows whose Processing Date is not {cutoff}")
    if source == "penalties":
        # A key that does not parse makes an event no label window can match: fail, never build it.
        # fine_amount and payment_denial_start_date may be missing (a fine has no denial date), not unparseable.
        broken = {c: invalid[c] for c in PENALTY_FIELDS if c in invalid}
        broken |= {
            c: typed[c].null_count()
            for c in ("penalty_date", "penalty_type")
            if typed[c].null_count()
        }
        if broken:
            raise PartitionError(f"penalty rows with a missing or unparseable key field: {broken}")
    stats: dict[str, Any] = {"rows": typed.height, "invalid": invalid}
    typed = _derive(typed, source, cfg, stats, frame)
    if source in EVENT_DATES:
        stats["post_cutoff_rows"] = typed.filter(pl.col(EVENT_DATES[source]) > cutoff).height

    outputs = {}
    for table in cfg.tables_for(source):
        out = typed.select(list(table.columns)).with_columns(
            pl.lit(vintage).alias("vintage_month"), pl.lit(cutoff).alias("cutoff_date")
        )
        _check_key(out, table)
        outputs[table.name] = out.sort(list(table.key), nulls_last=True)
    stats["non_null"] = {
        name: {c: out.height - out[c].null_count() for c in out.columns}
        for name, out in outputs.items()
    }
    return outputs, stats, res


def _derive(
    typed: pl.DataFrame, source: str, cfg: SilverConfig, stats: dict[str, Any], bronze: pl.DataFrame
) -> pl.DataFrame:
    if source == "provider_info" and "ownership_type" in typed.columns:
        typed = typed.with_columns(
            ownership_group(cfg.value_maps).alias("ownership_group"),
            pl.col("sprinkler_status__na").alias("sprinkler_status_not_available"),
        )
    elif source == "health_citations":
        typed = typed.with_columns(
            (pl.col("deficiency_prefix") + pl.col("deficiency_tag")).alias("ftag")
        )
    elif source == "penalties":
        # Identical rows are distinct fines (spec §3.5): number them in file order, never drop them.
        typed = typed.with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int64)
            .over(list(PENALTY_FIELDS))
            .alias("dup_seq")
        )
        typed = typed.with_columns(
            pl.concat_str(
                [pl.col(c).cast(pl.String).fill_null("") for c in (*PENALTY_FIELDS, "dup_seq")],
                separator="|",
            ).alias("penalty_event_key")
        )
        data = bronze.select(
            [c for c in bronze.columns if not c.startswith("_") and c != "vintage"]
        )
        stats["exact_duplicate_rows"] = data.height - data.unique().height
        stats["dup_seq_gt_1"] = typed.filter(pl.col("dup_seq") > 1).height
    elif source in ("mds_qm", "claims_qm"):
        start, end = measure_period_bounds(source)
        typed = typed.with_columns(
            start.alias("measure_period_start"), end.alias("measure_period_end")
        )
        bad = typed.filter(
            pl.col("measure_period").is_not_null() & pl.col("measure_period_end").is_null()
        )
        if bad.height:
            stats["invalid"]["measure_period"] = bad.height
    return typed


def _check_key(out: pl.DataFrame, table: SilverTable) -> None:
    dupes = out.select(list(table.key)).is_duplicated().sum()
    if dupes:
        raise PartitionError(f"{table.name}: {dupes} rows share a key {list(table.key)}")
    nulls = out.select(list(table.key)).null_count().row(0, named=True)
    if any(nulls.values()):
        raise PartitionError(f"{table.name}: null key values {nulls}")


def build_partition(
    bronze_sink: TableSink,
    silver_sink: TableSink,
    meta: MetadataStore,
    bronze: Mapping[str, Any],
    cfg: SilverConfig,
    *,
    run_id: str,
    now: datetime,
) -> PartitionResult:
    """Skip, build or fail the silver outputs of one bronze partition (its latest published bronze_files row).
    A failure writes nothing and gets one `failed` row per output table; the next run retries it."""
    source, vintage = bronze["table_name"], bronze["vintage_month"]
    tables = [t.name for t in cfg.tables_for(source)]
    if all(is_current(meta, silver_sink, t, bronze, cfg.config_hash) for t in tables):
        return PartitionResult(source, vintage, SKIPPED)

    base = {
        "run_id": run_id,
        "source_table": source,
        "vintage_month": vintage,
        "content_hash": bronze["content_hash"],
        "zip_sha256": bronze["zip_sha256"],
        "config_hash": cfg.config_hash,
        "built_at": now.isoformat(timespec="seconds"),
    }
    frame = bronze_sink.read(source, {"vintage": vintage})
    try:
        outputs, stats, _ = harmonize(frame, source, vintage, cfg)
    except PartitionError as exc:
        problems = resolve_headers(
            cfg.sources[source],
            [c for c in frame.columns if not c.startswith("_") and c != "vintage"],
            vintage,
        ).problems
        log.warning("silver failed for %s %s: %s", source, vintage, exc)
        meta.append(
            "silver_partitions",
            [base | {"table_name": t, "status": FAILED, "detail": str(exc)} for t in tables],
        )
        return PartitionResult(source, vintage, FAILED, str(exc), problems)

    rows = []
    for name, out in outputs.items():
        uri = silver_sink.write_partition(name, {"vintage": vintage}, out, overwrite=True)
        table_stats = {k: v for k, v in stats.items() if k != "non_null"} | {
            "non_null": stats["non_null"][name]
        }
        rows.append(
            base
            | {
                "table_name": name,
                "status": BUILT,
                "row_count": out.height,
                "stats": json.dumps(table_stats, sort_keys=True),
                "object_uri": uri,
            }
        )
    meta.append("silver_partitions", rows)
    log.info("silver built %s %s (%d rows)", source, vintage, stats["rows"])
    return PartitionResult(source, vintage, BUILT, stats=stats)
