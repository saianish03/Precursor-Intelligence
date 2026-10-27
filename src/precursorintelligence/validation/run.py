"""One validate run over a layer: read each table's months, check them, and record what was found.

Writes, all keyed by the run id:
  reports/validation/run=<id>/stats_<table>.json   the statistics of every month it looked at
  reports/validation/run=<id>/anomalies.json       every anomaly, severity error (schema) or warning (drift)
  metadata/validation/run_date=<UTC date>/validation_run_<id>_<ts>.json   the run manifest
  dq_results                                        one row per anomaly
It never writes to the layer it reads, takes no lock, and runs no freshness check.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..common.config import ConfigError, ValidationConfig, load_splits_config
from ..gold.dataset import MANIFEST_NAME
from ..ingestion.context import Context
from ..ingestion.pipeline import EXIT_OK, EXIT_VALIDATION
from .schema import TableSchema, infer_schema, load_schema, write_schema
from .stats import table_stats
from .tables import TableReader
from .validate import ERROR, WARNING, Anomaly, check_drift, check_month, check_schema

log = logging.getLogger(__name__)

NODE = "validation"


@dataclass
class ValidationResult:
    run_id: str
    layer: str
    strict: bool
    checked: dict[str, list[str]] = field(default_factory=dict)  # table -> months checked
    anomalies: list[Anomaly] = field(default_factory=list)
    report_prefix: str = ""
    manifest_key: str = ""

    def count(self, severity: str) -> int:
        return sum(a.severity == severity for a in self.anomalies)

    @property
    def exit_code(self) -> int:
        failing = self.count(ERROR) + (self.count(WARNING) if self.strict else 0)
        return EXIT_VALIDATION if failing else EXIT_OK


def _tables(cfg: ValidationConfig, layer: str, tables: list[str] | None) -> list[str]:
    if layer not in cfg.layers:
        raise ConfigError(f"unknown layer {layer!r}; configured: {sorted(cfg.layers)}")
    known = cfg.layers[layer].tables
    unknown = sorted(set(tables or []) - set(known))
    if unknown:
        raise ConfigError(f"{layer} has no configured table {unknown}; configured: {known}")
    return tables or known


def _reader(ctx: Context, cfg: ValidationConfig, layer: str, table: str) -> TableReader:
    return TableReader(ctx.object_store, layer, cfg.layers[layer], table)  # type: ignore[index]


def write_schemas(
    ctx: Context, cfg: ValidationConfig, *, layer: str, tables: list[str] | None = None
) -> list[Path]:
    """Infer each table's schema from all its months and (over)write configs/contracts/<table>.yaml."""
    written: list[Path] = []
    for table in _tables(cfg, layer, tables):
        reader = _reader(ctx, cfg, layer, table)
        months = reader.months()
        if not months:
            raise ConfigError(f"{layer}/{table} has no months to infer a schema from")
        schema = infer_schema(
            ((m, reader.month(m)) for m in months),
            table=table,
            layer=layer,
            low_cardinality_max=cfg.low_cardinality_max,
        )
        path = cfg.schema_path(table)
        write_schema(schema, path)
        log.info("wrote schema %s from %d months", path, len(months))
        written.append(path)
    return written


def _spine_start(ctx: Context) -> str:
    """The spine_start gold was built with (``gold --splits-config`` may differ from configs/splits.yaml): from
    ``gold/dataset_manifest.json``, else configs/splits.yaml when there is no manifest."""
    key = f"gold/{MANIFEST_NAME}"
    if ctx.object_store.exists(key):
        with ctx.object_store.open_read(key) as f:
            return str(json.load(f)["spine_start"])
    return load_splits_config().spine_start


def _to_check(months: list[str], month: str | None, all_months: bool, table: str) -> list[str]:
    if month is not None:
        if month not in months:
            raise ConfigError(f"{table} has no month {month}; it has {months[0]}..{months[-1]}")
        return [month]
    return months if all_months else months[-1:]


def _check_table(
    reader: TableReader,
    schema: TableSchema,
    cfg: ValidationConfig,
    months: list[str],
    checked: list[str],
    anomalies: list[Anomaly],
    first_month: str | None,
) -> dict[str, dict[str, Any]]:
    """Schema-check each month in ``checked`` and compare its stats with the month before's.
    Returns the stats computed, each month's once."""
    stats: dict[str, dict[str, Any]] = {}

    def stats_of(m: str) -> dict[str, Any]:
        if m not in stats:
            stats[m] = table_stats(
                reader.month(m), table=reader.table, month=m, top_values=cfg.top_values
            )
        return stats[m]

    for m in checked:
        df = reader.month(m)
        anomalies += check_schema(df, schema, month=m)
        anomalies += check_month(df, table=reader.table, month=m, first_month=first_month)
        i = months.index(m)
        current = stats_of(m)
        if i > 0:
            anomalies += check_drift(
                current, stats_of(months[i - 1]), cfg.drift, table=reader.table
            )
    return stats


def run_validation(
    ctx: Context,
    cfg: ValidationConfig,
    *,
    layer: str,
    tables: list[str] | None = None,
    month: str | None = None,
    all_months: bool = False,
    strict: bool = False,
) -> ValidationResult:
    """Check the latest month of each table (or ``month``, or every month: ``all_months`` or the layer's
    ``all_months`` setting) and record the results."""
    if month is not None and all_months:
        raise ConfigError("give --month or --all-months, not both")
    names = _tables(cfg, layer, tables)
    # every schema file first: a missing or broken one is a config error (exit 3) before any work
    schemas = {t: load_schema(cfg.schema_path(t)) for t in names}
    # gold has no rows before the spine_start it was built with; silver keeps its 2021-2022 history
    first_month = _spine_start(ctx) if layer == "gold" else None
    result = ValidationResult(run_id=ctx.run_id, layer=layer, strict=strict)
    result.report_prefix = f"reports/validation/run={ctx.run_id}"
    stats_docs: dict[str, dict[str, Any]] = {}
    for table in names:
        reader = _reader(ctx, cfg, layer, table)
        months = reader.months()
        if not months:
            result.anomalies.append(
                Anomaly(table, "", "table_missing", ERROR, None, None, f"{layer}/{table}",
                        f"{layer}/{table} has no months in the store")
            )  # fmt: skip
            result.checked[table] = []
            continue
        every = all_months or cfg.layers[layer].all_months  # type: ignore[index]
        checked = _to_check(months, month, every, table)
        stats = _check_table(
            reader, schemas[table], cfg, months, checked, result.anomalies, first_month
        )
        result.checked[table] = checked
        stats_docs[table] = {
            "layer": layer,
            "table": table,
            "months": [stats[m] for m in sorted(stats)],
        }
        log.info("validated %s/%s: %d months", layer, table, len(checked))
    _record(ctx, result, stats_docs)
    return result


def _put_json(ctx: Context, key: str, doc: dict[str, Any]) -> None:
    data = json.dumps(doc, indent=2, sort_keys=False, default=str).encode("utf-8")
    ctx.object_store.put_bytes(key, data, overwrite=True)


def _record(ctx: Context, result: ValidationResult, stats_docs: dict[str, dict[str, Any]]) -> None:
    now = datetime.now(UTC)
    for table, doc in stats_docs.items():
        _put_json(ctx, f"{result.report_prefix}/stats_{table}.json", doc)
    counts = {ERROR: result.count(ERROR), WARNING: result.count(WARNING)}
    _put_json(
        ctx,
        f"{result.report_prefix}/anomalies.json",
        {
            "run_id": result.run_id,
            "layer": result.layer,
            "strict": result.strict,
            "checked": result.checked,
            "counts": counts,
            "anomalies": [a.to_dict() for a in result.anomalies],
        },
    )
    checked_at = now.isoformat(timespec="seconds")
    ctx.metadata.append(
        "dq_results",
        [
            {
                "run_id": result.run_id,
                "node": NODE,
                "layer": result.layer,
                "table_name": a.table,
                "vintage_month": a.month or None,
                "check_name": a.check,
                "column_name": a.column,
                "severity": a.severity,
                "observed": json.dumps(a.observed, default=str),
                "expected": json.dumps(a.expected, default=str),
                "detail": a.detail,
                "checked_at": checked_at,
            }
            for a in result.anomalies
        ],
    )
    result.manifest_key = (
        f"metadata/validation/run_date={now:%Y-%m-%d}/"
        f"validation_run_{ctx.run_id}_{now:%Y%m%dT%H%M%S%fZ}.json"
    )
    _put_json(
        ctx,
        result.manifest_key,
        {
            "run_id": ctx.run_id,
            "started_at": ctx.started_at,
            "finished_at": checked_at,
            "exit_code": result.exit_code,
            "layer": result.layer,
            "strict": result.strict,
            "checked": result.checked,
            "counts": counts,
            "reports": result.report_prefix,
        },
    )
