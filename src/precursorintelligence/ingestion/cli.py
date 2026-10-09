"""Command line entry point: ``shingest`` (or ``python -m precursorintelligence``).

Examples:
  shingest step1                                   # full Step 1 on the local env
  shingest step1 --zip-dates 2021-01-27,2021-12-27 # a subset of archive zips
  shingest step1 --steps resolve,report            # rebuild the report from stored metadata
  shingest step1 --env aws                         # same pipeline, S3 + Postgres backends
  shingest catalog                                 # list what CMS currently advertises
"""

from __future__ import annotations

import logging
import shutil
from datetime import date
from pathlib import Path

import typer

from ..common.config import ConfigError, load_settings
from ..common.context import build_context, new_run_id
from .catalog import discover
from ..common.obs.logging import setup_logging
from .pipeline import STEPS, run_step1

app = typer.Typer(add_completion=False, help="Precursor Intelligence data ingestion (Step 1: CMS archive download + inventory)")
log = logging.getLogger("precursorintelligence.cli")


def _init(env: str, env_file: Path | None, log_level: str | None):
    try:
        settings = load_settings(env, env_path=env_file)
    except ConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(2) from None
    run_id = new_run_id()
    lc = settings.env.logging
    log_file = setup_logging(settings.resolve_path(lc.config), log_dir=settings.resolve_path(lc.log_dir),
                             run_id=run_id, console_format=lc.console_format, level=log_level)
    return settings, run_id, log_file


def _ship_log(ctx, log_file: Path) -> None:
    """Copy this run's JSON log next to its reports in the object store (works for local/S3/GCS)."""
    for h in logging.getLogger("precursorintelligence").handlers:
        h.flush()
    if log_file.exists():
        with log_file.open("rb") as f:
            ctx.object_store.put_stream(f"reports/step1/run={ctx.run_id}/run.jsonl", iter(lambda: f.read(1 << 20), b""),
                                        overwrite=True)


@app.command()
def step1(
    env: str = typer.Option("dev", help="environment config: configs/<env>.yaml (dev | prod)"),
    env_file: Path | None = typer.Option(None, help="explicit env yaml (overrides --env)"),
    steps: str = typer.Option(",".join(STEPS), help=f"comma-separated subset of {','.join(STEPS)}"),
    zip_dates: str | None = typer.Option(None, help="only these zip dates, e.g. 2021-01-27,2026-07-29"),
    limit: int | None = typer.Option(None, help="process at most N selected zips (oldest first)"),
    log_level: str | None = typer.Option(None, help="console/module log level override, e.g. DEBUG"),
) -> None:
    """Run Step 1: discover -> download -> extract -> inventory -> resolve -> report."""
    settings, run_id, log_file = _init(env, env_file, log_level)
    ctx = build_context(settings, run_id=run_id)
    dates = [date.fromisoformat(d.strip()) for d in zip_dates.split(",")] if zip_dates else None
    try:
        res = run_step1(ctx, steps=tuple(s.strip() for s in steps.split(",") if s.strip()), zip_dates=dates, limit=limit)
    finally:
        _ship_log(ctx, log_file)
    typer.echo(f"run {res.run_id}: selected={res.selected} downloads={res.downloads} "
               f"extracted_files={res.extracted} zips_inventoried={res.inventoried}")
    for v in res.reports.values():
        typer.echo(f"  report: {v}")
    if res.failures:
        typer.echo(f"{len(res.failures)} failures:", err=True)
        for f in res.failures[:20]:
            typer.echo(f"  - {f}", err=True)
        raise typer.Exit(1)


@app.command()
def catalog(env: str = typer.Option("dev"), env_file: Path | None = typer.Option(None)) -> None:
    """Fetch the CMS archive catalog and print what is advertised (no zips are downloaded)."""
    settings, run_id, _ = _init(env, env_file, None)
    ctx = build_context(settings, run_id=run_id)
    discover(ctx)
    rows = sorted(ctx.metadata.read("catalog", {"active": True}), key=lambda r: (r["zip_date"], r["type"]))
    width = shutil.get_terminal_size((120, 20)).columns
    for r in rows:
        typer.echo(f"{r['zip_date']}  {r['type']:<13} {r['size_bytes'] / 1e6:8.1f} MB  {r['zip_name']}"[:width])
    typer.echo(f"{len(rows)} active entries")


if __name__ == "__main__":  # pragma: no cover
    app()
