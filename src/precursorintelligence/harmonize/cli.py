"""Command line entry point: ``precursor-pipeline`` (or ``python -m precursorintelligence.harmonize``).

One command per DAG task (docs/DATA_PIPELINE_SPEC.md §15.3), all sharing the DAG run's --run-id:
  precursor-pipeline bronze --run-id R
  precursor-pipeline silver --run-id R

Exit codes follow the pipeline contract: 0 ok, 1 error (retried), 2 data or validation failure,
3 config or usage error. The DAG fails 2 and 3 without retry.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import typer

from ..bias.cli import bias_report
from ..common.config import ConfigError
from ..common.io.run_lock import RunLockHeldError, RunLockNotHeldError
from ..gold.cli import gold
from ..ingestion.cli import ENV_FILE_OPT, ENV_OPT, RUN_ID_OPT, _init, _ship_log
from ..ingestion.context import build_context
from ..ingestion.pipeline import EXIT_CONFIG, EXIT_ERROR, EXIT_OK, EXIT_VALIDATION
from .columns import load_silver_config
from .pipeline import run_bronze
from .silver_pipeline import CHECK_NAME, run_silver
from .verify import verify_bronze

app = typer.Typer(
    add_completion=False,
    help="Precursor Intelligence pipeline nodes after ingestion (bronze, then silver and gold)",
)
app.command()(gold)  # the gold node; its code lives in src/precursorintelligence/gold/
app.command("bias-report")(bias_report)  # data bias slicing on gold (src/precursorintelligence/bias/)


@app.callback()
def root() -> None:
    """Keeps `bronze` a named sub-command, as silver and gold will be (a lone command would collapse)."""


@app.command()
def bronze(
    run_id: str | None = RUN_ID_OPT,
    env: str = ENV_OPT,
    env_file: Path | None = ENV_FILE_OPT,
    log_level: str | None = typer.Option(None, help="override the configured log level"),
    verify: bool = typer.Option(
        False,
        help="read-only: report what bronze holds against the inventory (no lock, nothing written)",
    ),
) -> None:
    """Load every selected CMS file into lossless all-text bronze Parquet (spec §3.1)."""
    orchestrated = run_id is not None
    settings, run_id, log_file = _init(env, env_file, log_level, run_id, label="bronze")
    ctx = build_context(settings, run_id=run_id)
    if verify:
        report = verify_bronze(ctx)
        typer.echo(report.render())
        raise typer.Exit(EXIT_OK if report.ok else EXIT_VALIDATION)
    try:
        res = run_bronze(ctx, own_lock=not orchestrated)
    except RunLockHeldError as exc:  # another run is busy: retry later
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_ERROR) from None
    except RunLockNotHeldError as exc:  # DAG wiring: the lock task never ran
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    finally:
        _ship_log(ctx, log_file, node="bronze")
    counts = Counter(f.status for f in res.files)
    summary = " ".join(
        f"{status}={counts[status]}" for status in ("loaded", "reloaded", "failed", "skipped")
    )
    typer.echo(f"run {res.run_id}: {summary} stale_refresh={res.stale_vintages}")
    typer.echo(f"  manifest: {res.manifest_key}")
    if res.failures:
        typer.echo(f"{len(res.failures)} failures:", err=True)
        for f in res.failures[:20]:
            typer.echo(f"  - {f}", err=True)
        raise typer.Exit(res.exit_code)


@app.command()
def silver(
    run_id: str | None = RUN_ID_OPT,
    env: str = ENV_OPT,
    env_file: Path | None = ENV_FILE_OPT,
    log_level: str | None = typer.Option(None, help="override the configured log level"),
    calibrate: bool = typer.Option(
        False,
        "--calibrate/--no-calibrate",
        help="also write the Checkpoint 3-4 proposals (posting lag, QM selection); backfill or on demand",
    ),
) -> None:
    """Harmonise bronze into typed silver tables, build the ledgers and universe tables, and write the
    Checkpoint 2 report (spec §3.2-§6); with --calibrate also the Checkpoint 3-4 proposals."""
    orchestrated = run_id is not None
    settings, run_id, log_file = _init(env, env_file, log_level, run_id, label="silver")
    ctx = build_context(settings, run_id=run_id)
    try:
        cfg = load_silver_config()
        res = run_silver(ctx, own_lock=not orchestrated, cfg=cfg, calibrate=calibrate)
    except ConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    except RunLockHeldError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_ERROR) from None
    except RunLockNotHeldError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    finally:
        _ship_log(ctx, log_file, node="silver")
    counts = Counter(p.status for p in res.partitions)
    summary = " ".join(f"{s}={counts[s]}" for s in ("built", "failed", "skipped"))
    typer.echo(
        f"run {res.run_id}: {summary} citation_lookup_versions={res.lookup_versions} "
        f"{CHECK_NAME}={'ok' if not res.missing else len(res.missing)}"
    )
    for name, rows in res.tables.items():
        typer.echo(f"  {name}: {rows} rows")
    typer.echo(
        f"  report: {', '.join(k for k in res.report_keys if '/latest/' in k and k.endswith('.md'))}"
    )
    typer.echo(f"  manifest: {res.manifest_key}")
    if res.failures:
        typer.echo(f"{len(res.failures)} failures:", err=True)
        for f in res.failures[:20]:
            typer.echo(f"  - {f}", err=True)
        raise typer.Exit(res.exit_code)


def main() -> None:
    """Console entry point. Click reports bad flags with exit code 2, which here means a data problem,
    so usage errors are mapped to the config code (3) instead."""
    try:
        code = app(standalone_mode=False)
    except typer.TyperException as exc:
        show = getattr(exc, "show", None)
        if callable(show):
            show()
        code = getattr(exc, "exit_code", EXIT_ERROR)
        code = EXIT_CONFIG if code == 2 else code
    except typer.Abort:
        code = EXIT_ERROR
    sys.exit(code if isinstance(code, int) else 0)


if __name__ == "__main__":  # pragma: no cover
    main()
