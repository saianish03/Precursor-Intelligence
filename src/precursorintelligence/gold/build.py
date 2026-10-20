"""The gold node's run: silver -> gold/spine and gold/labels (spec §7, §9, §11; SILVER_CONTRACT §8, §11).

1. Published vintages = the silver provider partitions (as silver itself counts them). V = the latest, or a
   validated ``--v``.
2. Labels are measured from the last vintage that is not a ``stale_refresh`` (``vintage_flags``, written by
   bronze); a stale V adds no events (spec §9.2). The spine still runs to V.
3. Ledger events first published after that vintage are dropped, so an earlier ``--v`` sees only what was
   published by then. (``final_scope_severity`` and ``removed_by_cms`` are still as of the latest silver.)
4. Everything is built and checked in ``gold/_staging/run=<id>/`` and copied to ``gold/`` only when every check
   passes; on a failure ``gold/`` is untouched. The staged files stay as the run's record (the object store
   only deletes under quarantine/ and tmp/).

Still to add once PR #7 is on dev: silver's freshness check, ``partitions_without_current_silver`` (contract §3).
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from ..common.config import ConfigError, LabelsConfig, SplitsConfig
from ..common.io.table_sink import ParquetTableSink
from ..ingestion.context import Context
from ..ingestion.pipeline import EXIT_OK, EXIT_VALIDATION
from . import GoldCheckError
from .labels import build_labels
from .spine import build_spine

SILVER_INPUTS = ("facility_universe", "facility_presence", "ledger_citations", "ledger_penalties")
# spec §0.5: vintages from 2021-01; project scope is 2021-2026
VINTAGE_RANGE = ("2021-01", "2026-12")
STALE_REFRESH = "stale_refresh"
_YYYY_MM = re.compile(r"\d{4}-(0[1-9]|1[0-2])")


@dataclass
class GoldResult:
    run_id: str
    v: str = ""
    v_labels: str = ""
    rows: dict[str, int] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)  # name -> sha256 of the promoted file
    failures: list[str] = field(default_factory=list)
    manifest_key: str = ""

    @property
    def exit_code(self) -> int:
        return EXIT_VALIDATION if self.failures else EXIT_OK


def published_vintages(sink: ParquetTableSink) -> list[str]:
    """Vintages with a silver provider partition, ascending; never-published months have none."""
    return sorted(p["vintage"] for p in sink.list_partitions("provider_by_vintage"))


def resolve_v(published: list[str], requested: str | None, spine_start: str) -> str:
    """V: the latest published vintage, or ``requested`` if it is a valid published vintage (else exit 3)."""
    if not published:
        raise GoldCheckError("no silver/provider_by_vintage partitions: run silver first")
    outside = [p for p in published if not VINTAGE_RANGE[0] <= p <= VINTAGE_RANGE[1]]
    if outside:
        raise GoldCheckError(f"silver has vintages outside {VINTAGE_RANGE}: {outside}")
    if requested is None:
        return published[-1]
    if not _YYYY_MM.fullmatch(requested):
        raise ConfigError(f"--v {requested!r} is not a YYYY-MM month")
    if requested > published[-1]:
        raise ConfigError(f"--v {requested} is after the latest published vintage {published[-1]}")
    if requested not in published:
        raise ConfigError(f"--v {requested} was never published (no silver partition)")
    if requested < spine_start:
        raise ConfigError(f"--v {requested} is before spine_start {spine_start}")
    return requested


def label_vintage(
    flags: list[dict[str, Any]], published: list[str], v: str
) -> tuple[str, list[str]]:
    """The last vintage up to ``v`` that is not a stale refresh, and the stale ones skipped on the way.

    Stops (exit 2) instead of guessing when ``vintage_flags`` has no stale_refresh row for a month it needs.
    """
    stale = {r["vintage_month"]: bool(r["value"]) for r in flags if r["flag"] == STALE_REFRESH}
    skipped: list[str] = []
    for month in reversed([p for p in published if p <= v]):
        if month not in stale:
            raise GoldCheckError(
                f"vintage_flags has no stale_refresh row for {month}. Bronze writes it; point --env or "
                "--env-file at the metadata database that came with this silver (e.g. data/ops/metadata.db)"
            )
        if not stale[month]:
            return month, skipped
        skipped.append(month)
    raise GoldCheckError(f"every vintage up to {v} is flagged stale_refresh")


def events_published_by(ledger: pl.DataFrame, v: str) -> pl.DataFrame:
    """Ledger rows already published at vintage ``v`` (an event posted later must not label earlier rows)."""
    return ledger.filter(pl.col("first_seen_vintage") <= v)


def _parquet(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf, compression="zstd")
    return buf.getvalue()


def _read(ctx: Context, key: str) -> pl.DataFrame:
    if not ctx.object_store.exists(key):
        raise GoldCheckError(f"missing silver input: {key}")
    with ctx.object_store.open_read(key) as f:
        return pl.read_parquet(f)


def _run(
    ctx: Context, labels_cfg: LabelsConfig, splits_cfg: SplitsConfig, v: str | None
) -> GoldResult:
    result = GoldResult(run_id=ctx.run_id)
    published: list[str] = []
    stale: list[str] = []
    try:
        published = published_vintages(ParquetTableSink(ctx.object_store, prefix="silver"))
        result.v = resolve_v(published, v, splits_cfg.spine_start)
        result.v_labels, stale = label_vintage(
            ctx.metadata.read("vintage_flags", {"flag": STALE_REFRESH}), published, result.v
        )
        silver = {n: _read(ctx, f"silver/{n}/part-0.parquet") for n in SILVER_INPUTS}
        spine = build_spine(
            silver["facility_universe"],
            published_vintages=published,
            spine_start=splits_cfg.spine_start,
            v=result.v,
        )
        labels = build_labels(
            spine,
            ledger_citations=events_published_by(silver["ledger_citations"], result.v_labels),
            ledger_penalties=events_published_by(silver["ledger_penalties"], result.v_labels),
            facility_presence=silver["facility_presence"],
            cfg=labels_cfg,
            v=result.v_labels,
        )
        staged = _stage(ctx, {"spine": spine, "labels": labels})
        _promote(ctx, staged)
        result.rows = {"spine": spine.height, "labels": labels.height}
        result.outputs = {n: hashlib.sha256(b).hexdigest() for n, b in staged.items()}
    except GoldCheckError as exc:
        result.failures.append(str(exc))
    return result


def _stage(ctx: Context, tables: dict[str, pl.DataFrame]) -> dict[str, bytes]:
    """Write to gold/_staging/run=<id>/, read back and check there; returns the checked bytes."""
    staged: dict[str, bytes] = {}
    for name, df in tables.items():
        data = staged[name] = _parquet(df)
        ctx.object_store.put_bytes(_staging_key(ctx, name), data, overwrite=True)
    back = {n: _read(ctx, _staging_key(ctx, n)) for n in tables}
    spine, labels = back["spine"], back["labels"]
    if spine.select("ccn", "t").is_duplicated().any():
        raise GoldCheckError("staged spine: (ccn, t) is not unique")
    if not labels.select("ccn", "t").equals(spine.select("ccn", "t")):
        raise GoldCheckError("staged labels: not exactly the spine's rows, in the spine's order")
    return staged


def _staging_key(ctx: Context, name: str) -> str:
    return f"gold/_staging/run={ctx.run_id}/{name}/part-0.parquet"


def _promote(ctx: Context, staged: dict[str, bytes]) -> None:
    for name, data in staged.items():
        ctx.object_store.put_bytes(f"gold/{name}/part-0.parquet", data, overwrite=True)
