"""The ``validate`` sub-command of ``precursor-pipeline`` (registered in harmonize/cli.py).

  precursor-pipeline validate --layer silver                  # latest month of each silver table vs the month before
  precursor-pipeline validate --layer silver --all-months     # every month vs the month before
  precursor-pipeline validate --layer gold --table labels --month 2025-06
  precursor-pipeline validate --layer silver --write-schema   # (re)infer configs/contracts/<table>.yaml from all months

Read-only on the layer: no lock and no freshness check, so it runs on any silver or gold in the store.
Exit codes: 0 clean (warnings allowed), 2 schema violations (or any warning with --strict), 3 config or usage error
(unknown layer or table, a missing or broken schema file, a month the table does not have), 1 error.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ..common.config import ConfigError, load_validation_config
from ..ingestion.cli import ENV_FILE_OPT, ENV_OPT, RUN_ID_OPT, _init, _ship_log
from ..ingestion.context import build_context
from ..ingestion.pipeline import EXIT_CONFIG, EXIT_OK
from .run import NODE, run_validation, write_schemas
from .validate import ERROR, WARNING

_SHOWN = 20  # anomalies printed; all of them are in anomalies.json


def validate(
    layer: str = typer.Option(..., "--layer", help="silver or gold"),
    table: list[str] | None = typer.Option(
        None, "--table", help="only this table (repeatable); default: every table of the layer"
    ),
    month: str | None = typer.Option(
        None,
        "--month",
        help="check this month (YYYY-MM: vintage in silver, t in gold); default: the latest",
    ),
    all_months: bool = typer.Option(
        False, "--all-months", help="check every month against the one before"
    ),
    strict: bool = typer.Option(False, "--strict", help="drift warnings also exit 2"),
    write_schema: bool = typer.Option(
        False, "--write-schema", help="infer the schema files from all months instead of checking"
    ),
    config: Path | None = typer.Option(
        None, "--config", help="validation settings file; default: configs/validation.yaml"
    ),
    run_id: str | None = RUN_ID_OPT,
    env: str = ENV_OPT,
    env_file: Path | None = ENV_FILE_OPT,
    log_level: str | None = typer.Option(None, help="override the configured log level"),
) -> None:
    """Check a layer's tables against their schema files and the previous month's statistics."""
    settings, run_id, log_file = _init(env, env_file, log_level, run_id, label="validate")
    ctx = build_context(settings, run_id=run_id)
    try:
        cfg = load_validation_config(config)
        if write_schema:
            for path in write_schemas(ctx, cfg, layer=layer, tables=table):
                typer.echo(f"wrote {path}")
            raise typer.Exit(EXIT_OK)
        res = run_validation(
            ctx, cfg, layer=layer, tables=table, month=month, all_months=all_months, strict=strict
        )
    except ConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    finally:
        _ship_log(ctx, log_file, node=NODE)
    months = sum(len(m) for m in res.checked.values())
    typer.echo(
        f"run {res.run_id}: {layer} tables={len(res.checked)} months={months} "
        f"errors={res.count(ERROR)} warnings={res.count(WARNING)}"
    )
    typer.echo(f"  reports: {res.report_prefix}/")
    typer.echo(f"  manifest: {res.manifest_key}")
    for a in res.anomalies[:_SHOWN]:
        typer.echo(
            f"  - {a.severity} {a.table} {a.month} {a.column or '-'} {a.check}: {a.detail}",
            err=True,
        )
    if len(res.anomalies) > _SHOWN:
        typer.echo(f"  ... {len(res.anomalies) - _SHOWN} more in anomalies.json", err=True)
    raise typer.Exit(res.exit_code)
